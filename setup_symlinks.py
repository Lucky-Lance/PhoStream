#!/usr/bin/env python3
"""
Batch create symlinks and verify video integrity.

Steps:
1. Create top-level symlinks for all _2fps directories (stripping _2fps suffix)
2. Recursively create sub-directory symlinks
3. Fix Unicode encoding issues (NFD vs NFC for Pokémon's é)
4. Restore missing video files from tar.gz
5. Verify all video paths exist
"""
import os
import json
import tarfile
import unicodedata
import shutil
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Batch create symlinks and verify video integrity")
    parser.add_argument(
        "--videos-dir",
        default="./videos",
        help="Path to videos directory",
    )
    parser.add_argument(
        "--json-file",
        default="./dataset/phostream_annotations.json",
        help="Path to annotations JSON file",
    )
    parser.add_argument(
        "--tar-file",
        default="./videos/youtube_dl_reencoded_2fps.tar.gz",
        help="Path to tar.gz backup file",
    )
    return parser.parse_args()


# ========== Functions ==========

def create_top_level_symlinks(videos_dir):
    """Create top-level symlinks for _2fps directories."""
    print("[Step 1] Creating top-level symlinks...")
    count = 0
    for item in os.listdir(videos_dir):
        if item.endswith('_2fps') and os.path.isdir(os.path.join(videos_dir, item)):
            link_name = os.path.join(videos_dir, item[:-5])
            if os.path.lexists(link_name) and os.path.islink(link_name):
                os.remove(link_name)
            if not os.path.lexists(link_name):
                os.symlink(item, link_name)
                print(f"  ✓ {item[:-5]} -> {item}")
                count += 1
    return count


def create_recursive_symlinks(directory):
    """Recursively create symlinks for _2fps sub-directories (real dirs only)."""
    print("[Step 2] Creating recursive symlinks...")
    created, skipped, errors = 0, 0, 0

    for root, dirs, files in os.walk(directory, topdown=False):
        for dirname in dirs:
            if dirname.endswith('_2fps'):
                full_path = os.path.join(root, dirname)
                # Skip symlink directories
                if os.path.islink(full_path):
                    continue
                link_name = full_path[:-5]

                if os.path.lexists(link_name):
                    if os.path.islink(link_name):
                        os.remove(link_name)
                    else:
                        skipped += 1
                        continue

                rel_path = os.path.relpath(full_path, os.path.dirname(link_name))
                try:
                    os.symlink(rel_path, link_name)
                    created += 1
                except Exception as e:
                    print(f"  ✗ {link_name}: {e}")
                    errors += 1
    print(f"  created={created}, skipped={skipped}, errors={errors}")
    return created, skipped, errors


def fix_unicode_symlinks(videos_dir):
    """Fix Unicode encoding issues for Pokémon and similar special characters."""
    print("[Step 3] Fixing Unicode encoding issues...")
    fixed = 0

    # Check specific paths
    check_paths = [
        "youtube_dl_reencoded_2fps/downloads_travel_vlog_20250922_221854",
        "youtube_dl_reencoded/downloads_travel_vlog_20250922_221854"
    ]

    for check_path in check_paths:
        full_path = os.path.join(videos_dir, check_path)
        if not os.path.exists(full_path):
            continue

        # Find real directory (NFD encoding)
        actual_dir = None
        for item in os.listdir(full_path):
            item_path = os.path.join(full_path, item)
            if "Inside_JAPAN" in item and "_2fps" in item:
                # Only process real directories, not symlinks
                if os.path.isdir(item_path) and not os.path.islink(item_path):
                    item_bytes = item.encode('utf-8')
                    # NFD encoding signature: e + combining accent (\xcc\x81)
                    if b'\xcc\x81' in item_bytes:
                        actual_dir = item
                        print(f"  Found NFD directory: {item}")

        if actual_dir and "_2fps" in check_path:
            # Under _2fps directory, remove incorrect symlinks (pointing to same directory)
            for item in os.listdir(full_path):
                item_path = os.path.join(full_path, item)
                if "Inside_JAPAN" in item and not item.endswith('_2fps'):
                    if os.path.islink(item_path):
                        target = os.readlink(item_path)
                        # Check if pointing to same directory (relative path without /)
                        if '/' not in target or target.endswith('_2fps'):
                            os.remove(item_path)
                            print(f"  Removed incorrect symlink: {item}")
                            fixed += 1

            # Create correct symlink (pointing to real directory)
            link_dir = full_path.replace('_2fps', '')
            link_name = unicodedata.normalize('NFC', actual_dir[:-5])
            link_path = os.path.join(link_dir, link_name)

            # Remove existing symlinks
            for item in os.listdir(link_dir):
                if "Inside_JAPAN" in item and not item.endswith('_2fps'):
                    item_path = os.path.join(link_dir, item)
                    if os.path.islink(item_path):
                        os.remove(item_path)
                        print(f"  Removed old symlink: {item}")

            # Create new symlink
            rel_target = os.path.relpath(os.path.join(full_path, actual_dir), link_dir)
            if not os.path.lexists(link_path):
                os.symlink(rel_target, link_path)
                print(f"  ✓ Created symlink: {link_name} -> {rel_target}")
                fixed += 1

    return fixed


def check_all_videos(videos_dir, json_file):
    """Verify all video paths exist."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total, found, missing = 0, 0, []
    for item in data:
        video_path = item.get('video_path', '')
        if video_path:
            total += 1
            full_path = os.path.join(videos_dir, video_path)
            if os.path.exists(full_path):
                found += 1
            else:
                missing.append(video_path)
    return total, found, missing


def extract_missing_files(videos_dir, tar_file, missing_paths):
    """Restore missing files from tar.gz."""
    if not os.path.exists(tar_file):
        print(f"  ✗ tar.gz not found: {tar_file}")
        return 0

    print("  Extracting missing files from tar.gz...")
    restored = 0

    with tarfile.open(tar_file, 'r:gz') as tar:
        for missing in missing_paths:
            parts = missing.split('/')
            if len(parts) >= 4:
                # Build the tar-internal path
                tar_parts = parts.copy()
                tar_parts[0] = parts[0] + '_2fps'  # youtube_dl_reencoded_2fps
                tar_parts[-2] = parts[-2] + '_2fps'  # Name_reencoded_2fps
                tar_expected = '/'.join(tar_parts)

                for member in tar.getmembers():
                    # NFC normalize comparison (ignore encoding differences)
                    if unicodedata.normalize('NFC', member.name) == unicodedata.normalize('NFC', tar_expected):
                        try:
                            tar.extract(member, videos_dir)
                            print(f"  ✓ Restored: {member.name}")
                            restored += 1
                        except Exception as e:
                            print(f"  ✗ {member.name}: {e}")
    return restored


def fix_missing_symlinks(videos_dir, missing_paths):
    """Create symlinks for restored files."""
    print("  Creating symlinks...")
    fixed = 0

    for missing in missing_paths:
        parts = missing.split('/')
        if len(parts) >= 4:
            # Build the actual path
            actual_parts = parts.copy()
            actual_parts[0] = parts[0] + '_2fps'
            actual_parts[-2] = parts[-2] + '_2fps'
            actual_path = os.path.join(videos_dir, *actual_parts)

            if os.path.exists(actual_path):
                # Create symlink
                link_dir = os.path.join(videos_dir, *parts[:-1])
                link_name = parts[-2]  # Directory name (without _2fps)
                link_path = os.path.join(link_dir, link_name)

                if not os.path.exists(link_dir):
                    os.makedirs(link_dir, exist_ok=True)

                # Remove existing symlink
                if os.path.lexists(link_path) and os.path.islink(link_path):
                    os.remove(link_path)

                # Create new symlink
                rel_target = os.path.relpath(actual_path, link_dir)
                try:
                    os.symlink(rel_target, link_path)
                    print(f"  ✓ {link_name} -> {rel_target}")
                    fixed += 1
                except Exception as e:
                    print(f"  ✗ {link_name}: {e}")
    return fixed


# ========== Main ==========

def main(args):
    print("=" * 60)
    print("Batch Symlink Setup Script")
    print("=" * 60)
    print(f"  Videos dir: {args.videos_dir}")
    print(f"  JSON file:  {args.json_file}")
    print(f"  tar file:   {args.tar_file}")

    # Step 1: Top-level symlinks
    top_count = create_top_level_symlinks(args.videos_dir)
    print(f"  Top-level symlinks: {top_count}")

    # Step 2: Recursive symlinks
    create_recursive_symlinks(args.videos_dir)

    # Step 3: Unicode fix
    unicode_fixed = fix_unicode_symlinks(args.videos_dir)
    print(f"  Unicode fixes: {unicode_fixed}")

    # Step 4: Verify
    print("\n[Step 4] Verifying video integrity...")
    total, found, missing = check_all_videos(args.videos_dir, args.json_file)
    print(f"  total={total}, found={found}, missing={len(missing)}")

    # Step 5: Restore missing files
    if missing:
        print("\n[Step 5] Restoring missing files...")
        restored = extract_missing_files(args.videos_dir, args.tar_file, missing)

        if restored > 0:
            # Step 6: Create new symlinks
            print("\n[Step 6] Creating new symlinks...")
            create_recursive_symlinks(args.videos_dir)

            # Fix Unicode again
            print("\n[Step 7] Fixing Unicode again...")
            fix_unicode_symlinks(args.videos_dir)

        # Final verification
        print("\n[Step 8] Final verification...")
        total, found, missing = check_all_videos(args.videos_dir, args.json_file)

    # Results
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"Total videos: {total}")
    print(f"Found: {found}")
    print(f"Missing: {len(missing)}")

    if missing:
        print("\nMissing videos:")
        for p in missing[:5]:
            print(f"  - {p}")
        if len(missing) > 5:
            print(f"  ... and {len(missing)-5} more")
        return False
    else:
        print("\n✓ All video paths exist!")
        return True


if __name__ == "__main__":
    args = parse_args()
    success = main(args)
    exit(0 if success else 1)
