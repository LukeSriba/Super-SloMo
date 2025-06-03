import av
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from scipy.stats import entropy

def calculate_entropy(image_gray):
    """Calculate entropy of a grayscale image."""
    hist, _ = np.histogram(image_gray.flatten(), bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]  # avoid log(0)
    return entropy(hist, base=2)

def frame_to_gray(frame):
    """Convert PyAV frame (in RGB24) to grayscale numpy array."""
    img = frame.to_ndarray(format='gray')
    return img

def compare_videos(video_path1, video_path2, max_frames=None):
    container1 = av.open(video_path1)
    container2 = av.open(video_path2)

    frames1 = container1.decode(video=0)
    frames2 = container2.decode(video=0)

    psnr_values = []
    ssim_values = []
    entropy_orig_values = []
    entropy_proc_values = []

    count = 0
    while True:
        try:
            frame1 = next(frames1)
            frame2 = next(frames2)
        except StopIteration:
            break

        img1 = frame_to_gray(frame1)
        img2 = frame_to_gray(frame2)

        # Make sure frames are the same size
        if img1.shape != img2.shape:
            raise ValueError(f"Frame size mismatch at frame {count}: {img1.shape} vs {img2.shape}")

        psnr = compare_psnr(img1, img2, data_range=255)
        ssim = compare_ssim(img1, img2, data_range=255)

        entropy_orig = calculate_entropy(img1)
        entropy_proc = calculate_entropy(img2)

        psnr_values.append(psnr)
        ssim_values.append(ssim)
        entropy_orig_values.append(entropy_orig)
        entropy_proc_values.append(entropy_proc)

        count += 1
        if max_frames and count >= max_frames:
            break

    container1.close()
    container2.close()

    print(f"Processed {count} frames")
    print(f"Average PSNR: {np.mean(psnr_values):.4f}")
    print(f"Average SSIM: {np.mean(ssim_values):.4f}")
    print(f"Average Entropy Original: {np.mean(entropy_orig_values):.4f}")
    print(f"Average Entropy Processed: {np.mean(entropy_proc_values):.4f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python compare_videos.py original_video.mp4 processed_video.mp4 [max_frames]")
        sys.exit(1)
    original_video_path = sys.argv[1]
    processed_video_path = sys.argv[2]
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else None

    compare_videos(original_video_path, processed_video_path, max_frames)
