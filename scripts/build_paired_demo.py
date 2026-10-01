"""
Build Paired Forensic Demonstration Dataset.

Discovers identity-linked real↔fake video pairs from the FakeAVCeleb dataset
for the MediaDNA same-source demonstration requirement (Section 20 of V16 prompt).

For each identity in RealVideo-RealAudio, finds the corresponding manipulated
videos in FakeVideo-FakeAudio, FakeVideo-RealAudio, and RealVideo-FakeAudio.

Output:
    data/paired_demo.csv — CSV with columns:
        pair_id, identity, real_path, fake_path, fake_type, label
"""

import os
import csv
import sys
import argparse
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_ROOT = os.path.join(PROJECT_ROOT, "FakeAVCeleb_v1.2", "FakeAVCeleb_v1.2")

CATEGORIES = {
    "RealVideo-RealAudio": 0,  # label 0 = real
    "FakeVideo-FakeAudio": 1,  # label 1 = fake
    "FakeVideo-RealAudio": 1,  # label 1 = fake
    "RealVideo-FakeAudio": 1,  # label 1 = fake
}

FAKE_CATEGORIES = ["FakeVideo-FakeAudio", "FakeVideo-RealAudio", "RealVideo-FakeAudio"]


def discover_identities(category_dir):
    """Walk a category directory and return {identity: [video_paths]}."""
    identity_videos = defaultdict(list)

    if not os.path.exists(category_dir):
        print(f"  [SKIP] Directory not found: {category_dir}")
        return identity_videos

    for race_dir in os.listdir(category_dir):
        race_path = os.path.join(category_dir, race_dir)
        if not os.path.isdir(race_path) or race_dir.startswith('.') or race_dir == 'desktop.ini':
            continue

        for gender_dir in os.listdir(race_path):
            gender_path = os.path.join(race_path, gender_dir)
            if not os.path.isdir(gender_path):
                continue

            for identity_dir in os.listdir(gender_path):
                identity_path = os.path.join(gender_path, identity_dir)
                if not os.path.isdir(identity_path):
                    continue

                for video_file in os.listdir(identity_path):
                    if video_file.endswith('.mp4'):
                        rel_path = os.path.relpath(
                            os.path.join(identity_path, video_file),
                            PROJECT_ROOT
                        )
                        # Normalize path separators
                        rel_path = rel_path.replace('\\', '/')
                        identity_videos[identity_dir].append(rel_path)

    return identity_videos


def build_pairs(max_pairs_per_identity=3, max_total=50):
    """Build identity-linked real↔fake pairs."""
    print("=" * 60)
    print("BUILDING PAIRED FORENSIC DEMONSTRATION DATASET")
    print("=" * 60)

    # Step 1: Find all real videos by identity
    real_dir = os.path.join(DATASET_ROOT, "RealVideo-RealAudio")
    print(f"\n[1/2] Scanning real videos: {real_dir}")
    real_identities = discover_identities(real_dir)
    print(f"  Found {len(real_identities)} identities with real videos")

    # Step 2: For each fake category, find matching identities
    fake_by_category = {}
    for fake_cat in FAKE_CATEGORIES:
        fake_dir = os.path.join(DATASET_ROOT, fake_cat)
        print(f"\n[2/2] Scanning {fake_cat}: {fake_dir}")
        fake_identities = discover_identities(fake_dir)
        fake_by_category[fake_cat] = fake_identities
        print(f"  Found {len(fake_identities)} identities with {fake_cat} videos")

    # Step 3: Build pairs
    pairs = []
    pair_id = 0

    for identity, real_videos in sorted(real_identities.items()):
        if pair_id >= max_total:
            break

        real_video = real_videos[0]  # Use first real video for this identity

        for fake_cat in FAKE_CATEGORIES:
            if pair_id >= max_total:
                break

            fake_videos = fake_by_category[fake_cat].get(identity, [])
            if not fake_videos:
                continue

            for fake_video in fake_videos[:max_pairs_per_identity]:
                if pair_id >= max_total:
                    break

                pair_id_str = f"pair_{pair_id:03d}"

                # Add real entry
                pairs.append({
                    "pair_id": pair_id_str,
                    "identity": identity,
                    "video_path": real_video,
                    "counterpart_path": fake_video,
                    "type": "RealVideo-RealAudio",
                    "label": 0,
                    "role": "real",
                })

                # Add fake entry
                pairs.append({
                    "pair_id": pair_id_str,
                    "identity": identity,
                    "video_path": fake_video,
                    "counterpart_path": real_video,
                    "type": fake_cat,
                    "label": 1,
                    "role": "fake",
                })

                pair_id += 1

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Build Paired Demo Dataset")
    parser.add_argument("--max-pairs", type=int, default=30,
                        help="Maximum number of real-fake pairs to generate")
    parser.add_argument("--max-per-identity", type=int, default=2,
                        help="Maximum fake variants per identity per category")
    parser.add_argument("--output", type=str, default=os.path.join(PROJECT_ROOT, "data", "paired_demo.csv"))
    args = parser.parse_args()

    pairs = build_pairs(max_pairs_per_identity=args.max_per_identity, max_total=args.max_pairs)

    if not pairs:
        print("\n[ERROR] No pairs found! Check dataset path.")
        sys.exit(1)

    # Write CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    fieldnames = ["pair_id", "identity", "video_path", "counterpart_path", "type", "label", "role"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)

    n_pairs = len(set(p["pair_id"] for p in pairs))
    n_real = sum(1 for p in pairs if p["role"] == "real")
    n_fake = sum(1 for p in pairs if p["role"] == "fake")

    # Count unique fake types
    fake_types = defaultdict(int)
    for p in pairs:
        if p["role"] == "fake":
            fake_types[p["type"]] += 1

    print(f"\n{'=' * 60}")
    print(f"PAIRED DEMO DATASET BUILT SUCCESSFULLY")
    print(f"{'=' * 60}")
    print(f"  Output:          {args.output}")
    print(f"  Total pairs:     {n_pairs}")
    print(f"  Real videos:     {n_real}")
    print(f"  Fake videos:     {n_fake}")
    print(f"  Total entries:   {len(pairs)}")
    print(f"\n  Fake types:")
    for ft, count in sorted(fake_types.items()):
        print(f"    {ft}: {count}")

    # Also create a minimal 3-pair dataset for quick demo
    demo_pairs = [p for p in pairs if p["pair_id"] in ("pair_000", "pair_001", "pair_002")]
    demo_output = args.output.replace("paired_demo", "paired_demo_mini")
    with open(demo_output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(demo_pairs)
    print(f"\n  Mini demo set:   {demo_output} ({len(demo_pairs)} entries)")


if __name__ == "__main__":
    main()
