import glob
import pandas as pd
import cv2
import librosa
import argparse
from skimage.metrics import structural_similarity as ssim
from moviepy.editor import *
from mfcc import mfcc
from correlation import correlate
from videohash import VideoHash

def compare_ssim(vpath1, vpath2):
    capA = cv2.VideoCapture(vpath1)
    capB = cv2.VideoCapture(vpath2)

    t = capA.get(cv2.CAP_PROP_FRAME_COUNT)
    tssim = 0
    for i in range(1,11):
        capA.set(1,t/i)
        capB.set(1,t/i)
        success, image1 = capA.read()
        success, image2 = capB.read()
        if success:
            image1 = cv2.resize(image1, (1280, 640))
            image2 = cv2.resize(image2, (1280, 640))
            # image1= cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            tssim = tssim+ssim(image1, image2, channel_axis=2, data_range=255, multichannel=True)
    tssim = tssim/10
    # print(tssim)
    # if tssim >50:
    #     return True
    # return False
    return tssim


def get_hash(video_arr):
    video_dic = {}
    for tem in video_arr:
        video_dic[tem] = VideoHash(tem)
    return video_dic


def compare_duration(va, vb):
    video_A = cv2.VideoCapture(va)
    video_B = cv2.VideoCapture(vb)
    durationA = video_A.get(cv2.CAP_PROP_FRAME_COUNT)
    durationB = video_B.get(cv2.CAP_PROP_FRAME_COUNT)
    # print(durationA)
    # print(durationB)
    video_A.release()
    video_B.release()
    if durationA - durationB != 0:
        return False
    else:
        return True
def audio_compare(va, vb):
    video = VideoFileClip(va)
    video.audio.write_audiofile("va.mp3")
    video = VideoFileClip(vb)
    video.audio.write_audiofile("vb.mp3")
    if librosa.get_duration(filename="va.mp3") < 5 :
        return mfcc("va.mp3","vb.mp3")
    else:
        return mfcc("va.mp3","vb.mp3") and correlate("va.mp3","vb.mp3")

def video_compare(video_dic1, video_arr1, video_dic2, video_arr2):
    same_video_record = []
    for i in range(len(video_arr1)):
        for j in range(len(video_arr2)):
            #print(video_arr1[i] + ", " + video_arr2[j])
            #compare_ssim(video_arr1[i], video_arr2[j])
            if (video_dic1[video_arr1[i]] - video_dic2[video_arr2[j]]) <= 2 and compare_duration(video_arr1[i],video_arr2[j]):
                ssim = compare_ssim(video_arr1[i], video_arr2[j])
                audio_match =audio_compare(video_arr1[i],video_arr2[j])
                if ssim > 0.54 and (ssim > 0.59 or audio_match):
                    same_video_record.append([video_arr1[i], video_arr2[j],ssim,audio_match])
                    video_arr2.pop(j)
                    j =-1
                    break
    return same_video_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i ", "--source-path", help="source file")
    parser.add_argument("-o ", "--target-path", help="target file")
    args = parser.parse_args()

    source = args.source_path if args.source_path else None
    target = args.target_path if args.target_path else None
    if source == None or target == None:
        raise Exception('No path')

    video_arr1 = glob.glob(source+"*.mp4")
    video_arr2 = glob.glob(target+"*.mp4")
    video_dic1 = get_hash(video_arr1)
    video_dic2 = get_hash(video_arr2)
    # df = pd.DataFrame(video_dic1)
    # df.to_csv("hashdict" + source+".csv")
    # df = pd.DataFrame(video_dic2)
    # df.to_csv("hashdict" + target+".csv")
    df = pd.DataFrame(video_compare(video_dic1, video_arr1, video_dic2, video_arr2))
    df.to_csv("test.csv")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/