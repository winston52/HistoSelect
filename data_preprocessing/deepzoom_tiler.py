from multiprocessing import Process, JoinableQueue
import argparse
import os
import re
import shutil
import sys
import glob
import numpy as np
import math
import datetime
from unicodedata import normalize
from PIL import Image, ImageFilter, ImageStat, ImageDraw

# Handle large whole slide images
Image.MAX_IMAGE_PIXELS = None

import openslide
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator

class TileWorker(Process):
    """A child process that generates and writes tiles."""
    def __init__(self, queue, slidepath, tile_size, overlap, limit_bounds,
                 quality, threshold):
        Process.__init__(self, name='TileWorker')
        self.daemon = True
        self._queue = queue
        self._slidepath = slidepath
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._quality = quality
        self._threshold = threshold
        self._slide = None

    def run(self):
        self._slide = open_slide(self._slidepath)
        last_associated = None
        dz = self._get_dz()
        while True:
            data = self._queue.get()
            if data is None:
                self._queue.task_done()
                break
            associated, level, address, outfile = data
            if last_associated != associated:
                dz = self._get_dz(associated)
                last_associated = associated
            try:
                tile = dz.get_tile(level, address)
                # Filter background (edge detection)
                edge = tile.filter(ImageFilter.FIND_EDGES)
                edge = ImageStat.Stat(edge).sum
                edge = np.mean(edge)/(self._tile_size**2)
                
                if edge > self._threshold:
                    w, h = tile.size
                    if not (w == self._tile_size and h == self._tile_size):
                        tile = tile.resize((self._tile_size, self._tile_size))
                    tile.save(outfile, quality=self._quality)
            except Exception:
                pass
            self._queue.task_done()

    def _get_dz(self, associated=None):
        if associated is not None:
            image = ImageSlide(self._slide.associated_images[associated])
        else:
            image = self._slide
        return DeepZoomGenerator(image, self._tile_size, self._overlap,
                                limit_bounds=self._limit_bounds)

class DeepZoomImageTiler(object):
    """Handles generation of tiles for specific magnification levels."""
    def __init__(self, dz, temp_base, target_levels, mag_base, format, associated, queue):
        self._dz = dz
        self._temp_base = temp_base
        self._format = format
        self._associated = associated or 'slide'
        self._queue = queue
        self._processed = 0
        self._target_levels = target_levels
        self._mag_base = int(mag_base)

    def run(self):
        target_levels = [self._dz.level_count - i - 1 for i in self._target_levels]
        mag_list = [int(self._mag_base / 2**i) for i in self._target_levels]
        
        mag_idx = 0
        for level in range(self._dz.level_count):
            if level not in target_levels:
                continue
            
            tiledir = os.path.join(f"{self._temp_base}_files", str(mag_list[mag_idx]))
            os.makedirs(tiledir, exist_ok=True)
            
            cols, rows = self._dz.level_tiles[level]
            for row in range(rows):
                for col in range(cols):
                    tilename = os.path.join(tiledir, f'{col}_{row}.{self._format}')
                    if not os.path.exists(tilename):
                        self._queue.put((None, level, (col, row), tilename))
                    self._tile_done()
            mag_idx += 1

    def _tile_done(self):
        self._processed += 1
        count, total = self._processed, self._dz.tile_count
        if count % 100 == 0 or count == total:
            print(f"Tiling {self._associated}: wrote {count}/{total} tiles", 
                  end='\r', file=sys.stderr)
            if count == total:
                print(file=sys.stderr)

class DeepZoomStaticTiler(object):
    """Main controller for tiling a whole slide."""
    def __init__(self, slidepath, temp_base, mag_levels, base_mag, objective, format, tile_size, overlap,
                limit_bounds, quality, workers, threshold, slide_id):
        self._slide = open_slide(slidepath)
        self._temp_base = temp_base
        self._format = format
        self._tile_size = tile_size
        self._overlap = overlap
        self._mag_levels = mag_levels
        self._base_mag = base_mag
        self._objective = objective
        self._limit_bounds = limit_bounds
        self._queue = JoinableQueue(2 * workers)
        self._workers = workers
        self._slide_id = slide_id
        
        for _ in range(workers):
            TileWorker(self._queue, slidepath, tile_size, overlap,
                       limit_bounds, quality, threshold).start()

    def run(self):
        dz = DeepZoomGenerator(self._slide, self._tile_size, self._overlap, limit_bounds=self._limit_bounds)
        m1 = self._slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
        m2 = self._slide.properties.get('aperio.AppMag')
        mag_base = int(m1) if m1 else (int(float(m2)) if m2 else self._objective)
            
        first_level = int(math.log2(float(mag_base) / self._base_mag))
        target_levels = [i + first_level for i in self._mag_levels]
        target_levels.reverse()
        
        tiler = DeepZoomImageTiler(dz, self._temp_base, target_levels, mag_base, self._format, self._slide_id, self._queue)
        tiler.run()
        self._shutdown()

    def _shutdown(self):
        for _ in range(self._workers):
            self._queue.put(None)
        self._queue.join()

def generate_triple_thumbnails(slide_path, patches_dir, thumb_slide_dir, base_mag, patch_size=256, levels=0):
    """Generates original, patch-overlay, and white-background visualizations."""
    slide = openslide.OpenSlide(slide_path)
    thumbnail_size = (3000, 3000)
    
    # 1. Original Thumbnail
    original_thumb = slide.get_thumbnail(thumbnail_size)
    os.makedirs(thumb_slide_dir, exist_ok=True)
    original_thumb.save(os.path.join(thumb_slide_dir, "original_thumbnail.png"))
    
    # Setup for Drawing
    patch_over_slide = original_thumb.copy()
    patch_on_white = Image.new("RGB", original_thumb.size, (255, 255, 255))
    
    draw_slide = ImageDraw.Draw(patch_over_slide)
    draw_white = ImageDraw.Draw(patch_on_white)
    
    target_mag = base_mag / pow(2, levels)
    m1 = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    m2 = slide.properties.get('aperio.AppMag')
    mag_base = int(m1) if m1 else (int(float(m2)) if m2 else 20)
    mag_correction = int(mag_base / target_mag)

    for patch_file in os.listdir(patches_dir):
        if patch_file.endswith(('.jpeg', '.png')):
            coords = patch_file.split('.')[0].split('_')
            x, y = int(coords[0]), int(coords[1])
            rx, ry = x * patch_size * mag_correction, y * patch_size * mag_correction
            
            tx = int(rx * original_thumb.size[0] / slide.dimensions[0])
            ty = int(ry * original_thumb.size[1] / slide.dimensions[1])
            bx = tx + int(patch_size * mag_correction * original_thumb.size[0] / slide.dimensions[0])
            by = ty + int(patch_size * mag_correction * original_thumb.size[1] / slide.dimensions[1])

            draw_slide.rectangle([(tx, ty), (bx, by)], outline='blue', width=2)
            draw_white.rectangle([(tx, ty), (bx, by)], outline='blue', width=2)

    # 2. Patch over Slide
    patch_over_slide.save(os.path.join(thumb_slide_dir, "patch_over_slide.png"))
    # 3. Patch on White
    patch_on_white.save(os.path.join(thumb_slide_dir, "patch_on_white.png"))

def finalize_patches(output_base, slide_patch_dir, ext='png'):
    temp_dir = f"{output_base}_WSI_temp_files"
    patches = glob.glob(os.path.join(temp_dir, '*', f'*.{ext}'))
    for p in patches:
        shutil.move(p, os.path.join(slide_patch_dir, os.path.basename(p)))
    return len(patches)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Final Patch Extraction Script')
    parser.add_argument('-e', '--overlap', type=int, default=0)
    parser.add_argument('-f', '--format', type=str, default='png')
    parser.add_argument('-v', '--slide_format', type=str, default='svs')
    parser.add_argument('-j', '--workers', type=int, default=32)
    parser.add_argument('-q', '--quality', type=int, default=70)
    parser.add_argument('-s', '--tile_size', type=int, default=224)
    parser.add_argument('-b', '--base_mag', type=float, default=20)
    parser.add_argument('-m', '--magnifications', type=int, nargs='+', default=(0,))
    parser.add_argument('-o', '--objective', type=float, default=20)
    parser.add_argument('-t', '--background_t', type=int, default=15)
    parser.add_argument('-c', '--continue_g', type=str, default='False')
    parser.add_argument('--slide_path', type=str, required=True)
    parser.add_argument('--output_base', type=str, required=True)

    args = parser.parse_args()
    args.continue_g = args.continue_g.lower() == 'true'
    
    target_mag = int(args.base_mag / pow(2, args.magnifications[0]))
    mode = 'pyramid' if len(args.magnifications) > 1 else 'single'
    parent_folder = f"{mode}_b{target_mag}_t{args.background_t}"
    
    full_output_path = os.path.join(args.output_base, parent_folder)
    patch_root = os.path.join(full_output_path, 'Patch')
    thumb_root = os.path.join(full_output_path, 'Thumbnail')
    log_path = os.path.join(full_output_path, 'cutlog.txt')
    
    os.makedirs(patch_root, exist_ok=True)
    os.makedirs(thumb_root, exist_ok=True)

    # Log with Hours:Minutes:Seconds format
    hms_time = datetime.datetime.now().strftime('%H:%M:%S')
    
    with open(log_path, 'w') as log:
        log.write(f"Started: {hms_time}\n")
        log.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Threshold: {args.background_t}\n")
        log.write(f"Patch Size: {args.tile_size}\n")
        log.write(f"Base Magnification: {args.base_mag}\n")
        log.write("-" * 50 + "\n")

    all_slides = glob.glob(os.path.join(args.slide_path, f'*.{args.slide_format}'))
    cumulative_patch_count = 0
    slides_processed_count = 0
    
    for idx, slide_path in enumerate(all_slides):
        slide_filename = os.path.basename(slide_path)
        slide_id = os.path.splitext(slide_filename)[0]
        slide_patch_dir = os.path.join(patch_root, slide_id)
        slide_thumb_dir = os.path.join(thumb_root, slide_id)
        
        if args.continue_g and os.path.exists(slide_patch_dir) and os.listdir(slide_patch_dir):
            print(f"Skipping slide {idx+1}/{len(all_slides)}: {slide_id}")
            continue
        
        os.makedirs(slide_patch_dir, exist_ok=True)
        print(f"Processing slide {idx+1}/{len(all_slides)}: {slide_id}")
        
        temp_base = f"{args.output_base}_WSI_temp"
        if os.path.exists(f"{temp_base}_files"): shutil.rmtree(f"{temp_base}_files")
        
        DeepZoomStaticTiler(slide_path, temp_base, args.magnifications, args.base_mag, args.objective, 
                             args.format, args.tile_size, args.overlap, True, args.quality, 
                             args.workers, args.background_t, slide_id).run()
        
        patch_count = finalize_patches(args.output_base, slide_patch_dir, ext=args.format)
        
        cumulative_patch_count += patch_count
        slides_processed_count += 1
        avg_patches = cumulative_patch_count / slides_processed_count
        
        generate_triple_thumbnails(slide_path, slide_patch_dir, slide_thumb_dir, args.base_mag, 
                                   patch_size=args.tile_size, levels=args.magnifications[0])
        
        with open(log_path, 'a') as log:
            log.write(f"current slide id: {slide_filename}\n")
            log.write(f"current {float(target_mag)}x patches: {patch_count}\n")
            log.write(f"current avg {float(target_mag)}x patches: {avg_patches:.1f}\n")
            log.write("-" * 20 + "\n")
        
        if os.path.exists(f"{temp_base}_files"): shutil.rmtree(f"{temp_base}_files")

    print(f"\nAll tasks complete. Data in: {full_output_path}")