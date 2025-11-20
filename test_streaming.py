#!/usr/bin/env python
# coding: utf8
# Simple test script for streaming functionality

from os import path
import numpy as np
import bioread

# Use one of the test data files
DATA_PATH = path.join(path.dirname(path.abspath(__file__)), "test", "data")
TEST_FILE = path.join(DATA_PATH, "physio", "physio-4.4.0.acq")

print("Testing bioread streaming functionality...")
print("=" * 60)

# Test 1: Read entire file without streaming
print("\n1. Reading entire file (stream=False)...")
full_data = bioread.read(TEST_FILE, stream=False)
print(f"   Loaded {len(full_data.channels)} channels")
for i, ch in enumerate(full_data.channels):
    print(f"   Channel {i}: {ch.point_count} points, {ch.samples_per_second} Hz")

# Test 2: Read entire file with streaming enabled but no offset
print("\n2. Reading entire file with streaming (stream=True, start_sample=0, sample_count=None)...")
stream_data_full = bioread.read(TEST_FILE, stream=True, start_sample=0, sample_count=None)
print(f"   Loaded {len(stream_data_full.channels)} channels")
for i, ch in enumerate(stream_data_full.channels):
    print(f"   Channel {i}: {ch.point_count} points, {ch.samples_per_second} Hz")

# Test 3: Compare full reads
print("\n3. Comparing full read vs full stream read...")
all_match = True
for i in range(len(full_data.channels)):
    ch_full = full_data.channels[i]
    ch_stream = stream_data_full.channels[i]

    if ch_full.point_count != ch_stream.point_count:
        print(f"   ERROR: Channel {i} point counts don't match!")
        all_match = False
        continue

    if not np.array_equal(ch_full.raw_data, ch_stream.raw_data):
        print(f"   ERROR: Channel {i} data doesn't match!")
        all_match = False
    else:
        print(f"   Channel {i}: Data matches ✓")

if all_match:
    print("   All channels match! ✓")

# Test 4: Read partial file with streaming
start_sample = 100
sample_count = 500
print(f"\n4. Reading partial file (stream=True, start_sample={start_sample}, sample_count={sample_count})...")
stream_data_partial = bioread.read(TEST_FILE, stream=True, start_sample=start_sample, sample_count=sample_count)
print(f"   Loaded {len(stream_data_partial.channels)} channels")
for i, ch in enumerate(stream_data_partial.channels):
    print(f"   Channel {i}: {ch.point_count} points")

# Test 5: Compare partial stream with slice of full read
print("\n5. Comparing partial stream with slice of full read...")
all_match = True
for i in range(len(full_data.channels)):
    ch_full = full_data.channels[i]
    ch_stream = stream_data_partial.channels[i]
    div = ch_full.frequency_divider

    # Calculate the channel-specific start and count
    # start_sample and sample_count are in base rate, need to convert to channel rate
    # Use ceiling division to get the first sample at or after start_sample
    channel_start_sample = (start_sample + div - 1) // div  # Ceiling division
    base_end_sample = start_sample + sample_count
    channel_end_sample = (base_end_sample + div - 1) // div  # Ceiling division
    expected_channel_samples = channel_end_sample - channel_start_sample

    # Expected points for this channel
    expected_points = min(expected_channel_samples, ch_full.point_count - channel_start_sample)

    if ch_stream.point_count != expected_points:
        print(f"   ERROR: Channel {i} (div={div}) has {ch_stream.point_count} points, expected {expected_points}")
        all_match = False
        continue

    # Compare the raw data - use channel-specific indices
    full_slice = ch_full.raw_data[channel_start_sample:channel_start_sample + expected_points]
    stream_slice = ch_stream.raw_data

    if not np.array_equal(full_slice, stream_slice):
        print(f"   ERROR: Channel {i} (div={div}) partial data doesn't match!")
        print(f"          Full slice length: {len(full_slice)}, Stream slice length: {len(stream_slice)}")
        print(f"          Channel start sample: {channel_start_sample}, Expected points: {expected_points}")
        # Show first few values for debugging
        print(f"          Full first 5: {full_slice[:5]}")
        print(f"          Stream first 5: {stream_slice[:5]}")
        all_match = False
    else:
        print(f"   Channel {i} (div={div}): Partial data matches ✓")

if all_match:
    print("   All partial channels match! ✓")

# Test 6: Test time_index for partial stream
print("\n6. Testing time_index for partial stream...")
for i in range(min(3, len(stream_data_partial.channels))):  # Test first 3 channels
    ch_full = full_data.channels[i]
    ch_stream = stream_data_partial.channels[i]
    div = ch_full.frequency_divider

    if ch_stream.time_index is not None and ch_full.time_index is not None:
        # Calculate the channel-specific start sample (using ceiling division)
        channel_start_sample = (start_sample + div - 1) // div

        # The time_index should start from the correct offset
        expected_time_start = ch_full.time_index[channel_start_sample] if channel_start_sample < len(ch_full.time_index) else None

        if expected_time_start is not None:
            actual_time_start = ch_stream.time_index[0]
            if abs(expected_time_start - actual_time_start) < 0.001:  # Allow small floating point error
                print(f"   Channel {i} (div={div}): time_index starts correctly ({actual_time_start:.3f}s) ✓")
            else:
                print(f"   ERROR: Channel {i} (div={div}) time_index starts at {actual_time_start:.3f}s, expected {expected_time_start:.3f}s")
        else:
            print(f"   Channel {i} (div={div}): Skipping time_index check (start_sample beyond data)")

print("\n" + "=" * 60)
print("Streaming tests completed!")
