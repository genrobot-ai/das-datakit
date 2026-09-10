#!/bin/bash


PROTO_DIR="." 

# Function to modify .proto files
modify_proto_imports() {
    local directory=$PROTO_DIR
    
    find "$directory" -type f -name '*.proto' | while read -r file; do
        # Use sed to perform the replacements
        sed -i.bak -e 's|import "common/proto/common/header.proto";|import "common/header.proto";|' \
                   -e 's|import "common/proto/topic/\(.*\)";|import "common/\1";|' \
                   -e 's|import "common/\(.*\)";|import "\1";|' \
                   "$file"
        
        # Check if the file was modified
        if [[ -s "$file.bak" ]]; then
            echo "Modified: $file"
            # Remove the backup file if changes were made
            rm "$file.bak"
        fi
    done
}


OUTPUT_DIR="proto_gen"


rm  "$OUTPUT_DIR" -rf
mkdir -p "$OUTPUT_DIR"


npx pbjs -t static-module -w es6 -o "$OUTPUT_DIR/topic.js" *.proto
