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


def reader_for_streaming(io):
    """
    Read the headers of a file, return a Reader object that will allow you to
    stream the data in chunks with stream().
    """
    if not hasattr(io, 'read'):
        raise TypeError('{0} must be an opened file.'.format(io))
    if hasattr(io, 'encoding'):
        raise TypeError('{0} must be opened in binary mode'.format(io))
    return reader.Reader.read_headers(io)
