#!/usr/bin/python3

# correlation.py
import subprocess
import numpy
import os
import librosa
import math


# # seconds to sample audio file for
# sample_time = 500
# # minimum number of points that must overlap in cross correlation
# # exception is raised if this cannot be met
# min_overlap = 3


def calculate_fingerprints(filename):
    """
    calculate fingerprint
    Generate file.mp3.fpcalc by "fpcalc -raw -length 500 file.mp3"
    :param filename:
    :return: fingerprints
    """
    if os.path.exists(filename + '.fpcalc'):
        print("Found precalculated fingerprint for %s" % (filename))
        f = open(filename + '.fpcalc', "r")
        fpcalc_out = ''.join(f.readlines())
        f.close()
    else:
        print("Calculating fingerprint by fpcalc for %s" % (filename))
        fpcalc_out = str(subprocess.check_output(
            ['fpcalc', '-raw', '-length', str(librosa.get_duration(filename=filename)), filename])).strip().replace(
            '\\n', '').replace("'", "")
    fingerprint_index = fpcalc_out.find('FINGERPRINT=') + 12
    fingerprints = list(map(int, fpcalc_out[fingerprint_index:].split(',')))

    return fingerprints


def correlation(listx, listy):
    """
    returns correlation between lists
    :param listx:
    :param listy:
    :return:
    """
    if len(listx) == 0 or len(listy) == 0:
        # Error checking in main program should prevent us from ever being
        # able to get here.
        raise IndexError('Empty lists cannot be correlated.')
    if len(listx) > len(listy):
        listx = listx[:len(listy)]
    elif len(listx) < len(listy):
        listy = listy[:len(listx)]

    covariance = 0
    for i in range(len(listx)):
        covariance += 32 - bin(listx[i] ^ listy[i]).count("1")
    covariance = covariance / float(len(listx))

    return covariance / 32


def cross_correlation(list_x, list_y, offset):
    # return cross correlation, with listy offset from listx
    if offset > 0:
        list_x = list_x[offset:]
        list_y = list_y[:len(list_x)]
    elif offset < 0:
        offset = -offset
        list_y = list_y[offset:]
        list_x = list_x[:len(list_y)]
    return correlation(list_x, list_y)


def compare(list_x, list_y, span, step):
    """
        cross correlate list_x and list_y with offsets from -span to span
    :param list_x:
    :param list_y:
    :param span:
    :param step:
    :return:
    """
    if span > min(len(list_x), len(list_y)):
        # Error checking in main program should prevent us from ever being
        # able to get here.
        raise ValueError('span >= sample size: %i >= %i\n'
                         % (span, min(len(list_x), len(list_y)))
                         + 'Reduce span, reduce crop or increase sample_time.')
    corr_xy = []
    for offset in numpy.arange(-span, span + 1, step):
        corr_xy.append(cross_correlation(list_x, list_y, offset))
    return corr_xy


def max_index(list_x):
    """
    the max value in the list
    :param list_x:
    :return: return index of maximum value in list
    """
    index_of_max_value = 0
    max_value = list_x[0]
    for i, value in enumerate(list_x):
        if value > max_value:
            max_value = value
            index_of_max_value = i
    return index_of_max_value


def get_max_corr(corr, source, target, span, step, threshold):
    max_corr_index = max_index(corr)
    max_corr_offset = -span + max_corr_index * step
    if corr[max_corr_index] > threshold:
        print("File A: %s" % source)
        print("File B: %s" % target)
        print('Match with correlation of %.2f%% at offset %i'
              % (corr[max_corr_index] * 100.0, max_corr_offset))
        return True
    return False


def get_avg_corr(corr):
    print(sum(corr) / len(corr));


def correlate(source, target):
    fingerprint_source = calculate_fingerprints(source)
    fingerprint_target = calculate_fingerprints(target)
    # number of points to scan cross correlation over
    span = min(50, min(len(fingerprint_source), len(fingerprint_target)) - 1)
    # step size (in points) of cross correlation
    step = 1
    # report match when cross correlation has a peak exceeding threshold
    threshold = 0.55
    corr = compare(fingerprint_source, fingerprint_target, span, step)
    max_corr_offset = get_max_corr(corr, source, target, span, step, threshold)

    return max_corr_offset
