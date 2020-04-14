# -*- coding: utf-8 -*-
from subprocess import Popen, PIPE, STDOUT
import os
import eyed3
import errno

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
from numpy.lib import stride_tricks

from sliceSpectrogram import createSlicesFromSpectrograms
from audioFilesTools import isMono, getGenre
from config import rawDataPath
from config import spectrogramsPath
from config import pixelPerSecond

# Tweakable parameters
desiredSize = 128
originalSpecLog = []
# Define
currentPath = os.path.dirname(os.path.realpath(__file__))

# Remove logs
eyed3.log.setLevel("ERROR")


# Creates .png whole spectrograms from mp3 files
def createSpectrogramsFromAudio():
    genresID = dict()
    files = os.listdir(rawDataPath + 'wav/')
    files = [file for file in files if file.endswith(".wav")]
    nbFiles = len(files)

    # Create path if not existing
    if not os.path.exists(os.path.dirname(spectrogramsPath)):
        try:
            os.makedirs(os.path.dirname(spectrogramsPath))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # Rename files according to genre
    for index, filename in enumerate(files):
        name = filename.replace("wav", "mp3")
        print(name)
        print("Creating spectrogram for file {}/{}...".format(index + 1, nbFiles))
        fileGenre = getGenre(rawDataPath + 'mp3/' + name)
        genresID[fileGenre] = genresID[fileGenre] + 1 if fileGenre in genresID else 1
        fileID = genresID[fileGenre]
        newFilename = str(fileGenre) + "_" + str(fileID)
        plotstft(filename, newFilename)


# Whole pipeline .mp3 -> .png slices
def createSlicesFromAudio():
    print("Creating spectrograms...")
    createSpectrogramsFromAudio()
    print("Spectrograms created!")

    print("Creating slices...")
    createSlicesFromSpectrograms(desiredSize)
    print("Slices created!")


""" short time fourier transform of audio signal """


def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize / 2.0))), sig)
    # cols for windowing
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    frames = stride_tricks.as_strided(samples, shape=(int(cols), int(frameSize)),
                                      strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)


""" scale frequency axis logarithmically """


def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins - 1) / max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):], axis=1)
        else:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):int(scale[i + 1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins * 2, 1. / sr)[:freqbins + 1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i + 1])])]

    return newspec, freqs


""" plot spectrogram"""


def plotstft(audiopath, newFileName, binsize=2 ** 10):
    samplerate, samples = wavfile.read(rawDataPath + 'wav/' + audiopath)
    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    ims = 20. * np.log10(np.abs(sshow) / 10e-6)  # amplitude to decibel
    if 'original' in newFileName:
        originalSpecLog.append([audiopath.replace(".wav", ""), ims])
    else:
        # Taking 50 pixels per sec
        # if 'Berceuse' in audiopath: # Berceuse is approx 4 min 15 sec
        #     size = 127.5
        # else:                       # Fantaisie is approx 11 min 30 sec
        #     size = 375
        # originalSpecLog1 = [x[1] for x in originalSpecLog if x[0] in audiopath]
        # comp = originalSpecLog1[0]
        # originalSize = np.shape(comp)[0]
        # imsSize = np.shape(ims)[0]
        # if imsSize > originalSize:
        #     ims = ims[:originalSize]
        # elif imsSize < originalSize:
        #     comp = comp[:imsSize]
        # newims = comp - ims
        size = np.shape(np.transpose(ims))[1]/200
        plt.figure(figsize=(size, 1.28))
        plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap='gray', interpolation="none")
        plt.subplots_adjust(bottom=0)
        plt.subplots_adjust(top=1)
        plt.subplots_adjust(right=1)
        plt.subplots_adjust(left=0)
        plt.savefig(spectrogramsPath + newFileName.replace('.wav', '') + '.png')
