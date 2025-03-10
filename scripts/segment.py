import os
import shutil

def segment_video_frames(base_dir, segment_size=15):
    for video_folder in os.listdir(base_dir):
        video_path = os.path.join(base_dir, video_folder)

        if not os.path.isdir(video_path):
            continue 

        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith('.png')])

        for i in range(0, len(frame_files), segment_size):
            segment_name = f"{video_folder}_{i // segment_size + 1}"
            segment_path = os.path.join(video_path, segment_name)

            os.makedirs(segment_path, exist_ok=True)

            for frame_file in frame_files[i:i + segment_size]:
                src = os.path.join(video_path, frame_file)
                dst = os.path.join(segment_path, frame_file)
                shutil.move(src, dst)

            print(f"Segment {segment_name} created with {len(frame_files[i:i + segment_size])} frames.")

if __name__ == "__main__":
    base_directory = "./UVG"  # modify your downloaded dir here
    segment_video_frames(base_directory)