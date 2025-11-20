# coding: utf8
# Part of the bioread package for reading BIOPAC data.
#
# Copyright (c) 2023 Board of Regents of the University of Wisconsin System
#
# Written Nate Vack <njvack@wisc.edu> with research from John Ollinger
# at the Waisman Laboratory for Brain Imaging and Behavior, University of
# Wisconsin-Madison
# Project home: http://github.com/njvack/bioread

from __future__ import absolute_import

from warnings import deprecated

from bioread import reader

from ._metadata import version as __version__, author as __author__  # noqa


def read(filelike, channel_indexes=None, file_lock=None, bits=32, stream=False, start_sample=0, sample_count=None):
    """
    Read a file (either an IO object or a filename) and return a Datafile.

    channel_indexes:    A list of integer channel numbers. Other channels will
                        have empty data.
    file_lock: An optional lock used for multiprocessing
    bits: The number of bits channel data should be loaded in (16 bit, 32 bit or 64 bit)
    stream: Enable streaming mode to read only a portion of the file (default: False)
    start_sample: Starting sample position for streaming mode (default: 0)
    sample_count: Number of samples to read in streaming mode (default: None = all)
    """
    return reader.Reader.read(
        filelike,
        channel_indexes,
        file_lock=file_lock,
        bits=bits,
        stream=stream,
        start_sample=start_sample,
        sample_count=sample_count
    ).datafile


# Deprecated; provided for compatibility with previous versions.
read_file = read


def read_headers(filelike):
    """
    Read only the headers of a file, returns a Datafile with empty channels.
    """
    return reader.Reader.read_headers(filelike).datafile

@deprecated("Use read instead")
def reader_for_streaming(io):
    """
    Read the headers of a file, return a Reader object that allows multiple
    streaming operations without closing the file.

    The returned Reader object can be used to:
    - Call read_range() multiple times to read different portions of the file
    - Access file metadata via reader.datafile properties
    - Use stream() to get a chunk iterator

    Args:
        io: An opened file object in binary mode

    Returns:
        Reader object with headers loaded, file handle still open

    Raises:
        TypeError: If io is not an opened file or not in binary mode

    Example:
        with open('myfile.acq', 'rb') as f:
            reader = reader_for_streaming(f)
            # Read first 2 minutes
            data = reader.read_range(duration_seconds=120)
            # Read next minute
            data = reader.read_range(start_seconds=120, duration_seconds=60)

    Note: The file handle remains open and must be closed by the caller.
    """
    if not hasattr(io, 'read'):
        raise TypeError('{0} must be an opened file.'.format(io))
    if hasattr(io, 'encoding'):
        raise TypeError('{0} must be opened in binary mode'.format(io))
    return reader.Reader.read_headers(io)

@deprecated("Use read instead")
def read_initial_data(filelike, seconds=120, channel_indexes=None, bits=32):
    """
    Convenience function: Open a file, read the first N seconds of data,
    and return a Reader for additional operations.

    This is useful for the common pattern of loading initial data and then
    selectively loading more as needed.

    Args:
        filelike: Filename (string) or file-like object opened in binary mode
        seconds: Duration of initial data to read in seconds (default: 120 = 2 minutes)
        channel_indexes: List of channel indices to read (None = all channels)
        bits: Data precision in bits: 16, 32 (default), or 64

    Returns:
        Tuple of (reader, datafile):
            - reader: Reader object that can be used for additional read_range() calls
            - datafile: Datafile object containing the initial data

    Raises:
        TypeError: If file is compressed (streaming not supported)

    Example:
        # Read first 2 minutes, then read more as needed
        reader, initial_data = read_initial_data('myfile.acq', seconds=120)

        # Process initial data
        for channel in initial_data.channels:
            process(channel.data)

        # Read additional data from 5-6 minutes
        more_data = reader.read_range(start_seconds=300, duration_seconds=60)

        # Remember to close when done
        reader.acq_file.close()

    Tip: Use with context manager for automatic file closing:
        reader, data = read_initial_data('myfile.acq', seconds=120)
        try:
            process(data)
            more_data = reader.read_range(start_seconds=120, duration_seconds=60)
        finally:
            reader.acq_file.close()
    """
    # Open file if given a filename
    if isinstance(filelike, str):
        io = open(filelike, 'rb')
        should_keep_open = True
    else:
        io = filelike
        should_keep_open = False

    # Create reader and read headers
    rdr = reader_for_streaming(io)

    # Set the bits parameter
    rdr.bits = bits

    # Read the initial data range
    rdr.read_range(
        duration_seconds=seconds,
        channel_indexes=channel_indexes
    )

    return rdr, rdr.datafile
