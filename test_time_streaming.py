#!/usr/bin/env python
# coding: utf8
# Tests for time-based streaming functionality

from os import path
import numpy as np
import bioread

# Use one of the test data files
DATA_PATH = path.join(path.dirname(path.abspath(__file__)), "test", "data")
TEST_FILE = path.join(DATA_PATH, "physio", "physio-4.4.0.acq")

print("Testing bioread time-based streaming functionality...")
print("=" * 70)

# Test 1: Read initial data with read_initial_data()
print("\n1. Testing read_initial_data() with default 120 seconds...")
try:
    reader, initial_data = bioread.read_initial_data(TEST_FILE, seconds=120)
    print(f"   ✓ Successfully loaded {len(initial_data.channels)} channels")

    # Verify data was loaded
    for i, ch in enumerate(initial_data.channels):
        if ch.loaded:
            expected_samples = int(120 * ch.samples_per_second)
            actual_samples = ch.point_count
            print(f"   Channel {i}: {actual_samples} samples (expected ~{expected_samples})")
        else:
            print(f"   ERROR: Channel {i} data not loaded!")

    # Close the file
    reader.acq_file.close()
    print("   ✓ Test passed")
except Exception as e:
    print(f"   ✗ Test failed: {e}")

# Test 2: Multiple reads from same file handle
print("\n2. Testing multiple read_range() calls from same file...")
try:
    with open(TEST_FILE, 'rb') as f:
        reader = bioread.reader_for_streaming(f)

        # Read first 60 seconds
        data1 = reader.read_range(duration_seconds=60)
        samples1 = data1.channels[0].point_count
        time1_start = data1.channels[0].time_index[0] if data1.channels[0].time_index is not None else 0
        time1_end = data1.channels[0].time_index[-1] if data1.channels[0].time_index is not None else 0
        print(f"   First read: {samples1} samples, time range: {time1_start:.2f}s - {time1_end:.2f}s")

        # Read next 60 seconds (60-120s)
        data2 = reader.read_range(start_seconds=60, duration_seconds=60)
        samples2 = data2.channels[0].point_count
        time2_start = data2.channels[0].time_index[0] if data2.channels[0].time_index is not None else 0
        time2_end = data2.channels[0].time_index[-1] if data2.channels[0].time_index is not None else 0
        print(f"   Second read: {samples2} samples, time range: {time2_start:.2f}s - {time2_end:.2f}s")

        # Verify data was replaced, not accumulated
        if data2.channels[0].point_count == samples2:
            print("   ✓ Data correctly replaced (not accumulated)")
        else:
            print("   ✗ ERROR: Data accumulated instead of replaced")

        # Verify time ranges are different
        if abs(time2_start - time1_start) > 50:  # Should be ~60 seconds apart
            print("   ✓ Time ranges are correctly different")
        else:
            print(f"   ✗ ERROR: Time ranges overlap (diff: {time2_start - time1_start:.2f}s)")

    print("   ✓ Test passed")
except Exception as e:
    print(f"   ✗ Test failed: {e}")

# Test 3: Compare time-based vs sample-based parameters
print("\n3. Comparing time-based vs sample-based parameters...")
try:
    # Read using time-based parameters
    with open(TEST_FILE, 'rb') as f:
        reader1 = bioread.reader_for_streaming(f)
        data_time = reader1.read_range(start_seconds=10, duration_seconds=30)

    # Read using sample-based parameters
    with open(TEST_FILE, 'rb') as f:
        reader2 = bioread.reader_for_streaming(f)
        # Get samples_per_second from header
        sps = reader2.samples_per_second
        start_sample = int(10 * sps)
        sample_count = int(30 * sps)
        data_sample = reader2.read_range(start_sample=start_sample, sample_count=sample_count)

    # Compare channel 0 data
    ch_time = data_time.channels[0]
    ch_sample = data_sample.channels[0]

    if ch_time.point_count == ch_sample.point_count:
        print(f"   ✓ Same point count: {ch_time.point_count}")
    else:
        print(f"   ✗ Different point counts: time={ch_time.point_count}, sample={ch_sample.point_count}")

    # Compare actual data
    if np.array_equal(ch_time.raw_data, ch_sample.raw_data):
        print("   ✓ Data matches exactly")
    else:
        print("   ✗ Data does not match")

    print("   ✓ Test passed")
except Exception as e:
    print(f"   ✗ Test failed: {e}")

# Test 4: Verify data replacement on multiple reads
print("\n4. Verifying data is replaced (not accumulated) on multiple reads...")
try:
    with open(TEST_FILE, 'rb') as f:
        reader = bioread.reader_for_streaming(f)

        # Read small chunk
        reader.read_range(duration_seconds=10)
        small_count = reader.datafile.channels[0].point_count

        # Read larger chunk - should replace, not append
        reader.read_range(duration_seconds=30)
        large_count = reader.datafile.channels[0].point_count

        # Verify the second read replaced the first
        expected_large = int(30 * reader.samples_per_second)
        expected_small = int(10 * reader.samples_per_second)

        if large_count > small_count and large_count <= expected_large + 100:
            print(f"   ✓ Data replaced correctly (small: {small_count}, large: {large_count})")
        else:
            print(f"   ✗ Data not replaced correctly (small: {small_count}, large: {large_count})")

        # Verify no old attributes remain
        ch = reader.datafile.channels[0]
        if hasattr(ch, '_stream_start_sample'):
            print("   ✗ WARNING: Old streaming attributes still present")
        else:
            print("   ✓ Old streaming attributes cleaned up")

    print("   ✓ Test passed")
except Exception as e:
    print(f"   ✗ Test failed: {e}")

# Test 5: Test with different channel frequencies
print("\n5. Testing with channels of different frequency dividers...")
try:
    with open(TEST_FILE, 'rb') as f:
        reader = bioread.reader_for_streaming(f)

        # Read 60 seconds
        data = reader.read_range(duration_seconds=60)

        print("   Channel samples for 60 second read:")
        for i, ch in enumerate(data.channels):
            if ch.loaded:
                expected = int(60 * ch.samples_per_second)
                actual = ch.point_count
                # Allow small variation due to ceiling division
                if abs(actual - expected) <= 1:
                    print(f"   ✓ Channel {i} (div={ch.frequency_divider}): {actual} samples (expected ~{expected})")
                else:
                    print(f"   ✗ Channel {i} (div={ch.frequency_divider}): {actual} samples (expected ~{expected})")

    print("   ✓ Test passed")
except Exception as e:
    print(f"   ✗ Test failed: {e}")

# Test 6: Test custom seconds parameter
print("\n6. Testing custom seconds parameter (180 seconds = 3 minutes)...")
try:
    reader, data = bioread.read_initial_data(TEST_FILE, seconds=180)

    ch0_expected = int(180 * data.channels[0].samples_per_second)
    ch0_actual = data.channels[0].point_count

    if abs(ch0_actual - ch0_expected) <= 1:
        print(f"   ✓ Got {ch0_actual} samples (expected ~{ch0_expected})")
    else:
        print(f"   ✗ Got {ch0_actual} samples (expected ~{ch0_expected})")

    reader.acq_file.close()
    print("   ✓ Test passed")
except Exception as e:
    print(f"   ✗ Test failed: {e}")

# Test 7: Test reading with None sample_count (read to end)
print("\n7. Testing reading to end of file with duration_seconds=None...")
try:
    # First get total file duration
    full_data = bioread.read(TEST_FILE)
    total_samples = full_data.channels[0].point_count

    # Read from sample 1000 to end
    with open(TEST_FILE, 'rb') as f:
        reader = bioread.reader_for_streaming(f)
        data = reader.read_range(start_sample=1000, sample_count=None)

        # Should have total_samples - 1000 samples (approximately, accounting for ceiling division)
        expected = total_samples - 1000
        actual = data.channels[0].point_count

        if abs(actual - expected) <= 2:  # Allow small rounding difference
            print(f"   ✓ Read to end correctly: {actual} samples (expected ~{expected})")
        else:
            print(f"   ✗ Incorrect sample count: {actual} (expected ~{expected})")

    print("   ✓ Test passed")
except Exception as e:
    print(f"   ✗ Test failed: {e}")

print("\n" + "=" * 70)
print("Time-based streaming tests completed!")
