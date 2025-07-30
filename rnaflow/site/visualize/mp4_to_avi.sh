#!/bin/bash

search_path="path/to/your/mp4/files"  # Change this to the directory containing your MP4 files

find "$search_path" -type f -name "*.mp4" | while read -r mp4file; do
    dir=$(dirname "$mp4file")
    base=$(basename "$mp4file" .mp4)
    avifile="$dir/${base}.avi"
    
    echo "Converting: $mp4file to $avifile"
    
    ffmpeg -i "$mp4file" -pix_fmt nv12 -f avi -vcodec rawvideo "$avifile"
    
    if [ $? -eq 0 ]; then
        echo "Successfully converted: $mp4file"
    else
        echo "Failed to convert: $mp4file" >&2
    fi
done

echo "Conversion process completed."