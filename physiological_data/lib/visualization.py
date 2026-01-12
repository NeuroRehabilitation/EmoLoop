import matplotlib.pylab as plt
import matplotlib.image as mpimg
import numpy as np
import os
import argparse
import acquisition as acq
import tools


parser = argparse.ArgumentParser(description="Identify the path to the folder that contains all signals, photos and PsychoPy files.")
parser.add_argument("-P", "--path")
args = parser.parse_args()

if args.path:
    folder = fr"D:\Google Drive\Faculdade\Doutoramento\Acquisitions\new_biosignalsnotebooks\acquisitions\Acquisitions\{args.path}"
else:
    raise Exception("Please specify the path to the folder containing all signals, photos and PsychoPy files.")

acquisition = acq.Acquisition(folder)
acquisition.getBiosignalsSensors(acq.ALL, True)
biosignals = acquisition.signal
time_answer = [acquisition.psycho.getTimestamps(), acquisition.psycho.getTimeAnswer()]
time_axis, biosignals = biosignals[:, 0], biosignals[:, 1:]

print("Shape: ", biosignals.shape, np.where(acquisition.sensors == acq.ACC)[0])

pictures_filenames = []
timestamps = []
for name in os.listdir(folder):
    if 'ip_' in name and '-acquisition' in name:
        folder_snapshots = os.path.join(os.path.join(folder, name), 'snapshot')
        for filename in os.listdir(folder_snapshots):
            if filename.endswith('.jpeg'): 
                pictures_filenames.append(filename)
                timestamps.append(float(filename.split('.jpeg')[0]))

index = tools.closestIndex(timestamps, time_answer[0][0])
timestamps = np.array(timestamps)[np.where(timestamps > time_answer[0][0])[0]]


plt.ion()
fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(6, 12)

ax1 = fig.add_subplot(gs[0, 0:3])
ax1.set_title('fNIRS 1')
acquisition.getBiosignalsSensors([acq.FNIRS], True)
ax1.plot(time_axis, acquisition.signal[:, 1])

ax2 = fig.add_subplot(gs[0, 3:6], sharex=ax1)
ax2.set_title('fNIRS 2')
ax2.plot(time_axis, acquisition.signal[:, 2])

ax3 = fig.add_subplot(gs[0:3, 6:12])
img = mpimg.imread(os.path.join(folder_snapshots, pictures_filenames[index]))
picture = ax3.imshow(img)
ax3.axis('off')

ax4 = fig.add_subplot(gs[1, :3], sharex=ax1)
ax4.set_title('fNIRS 3')
acquisition.getBiosignalsSensors([acq.FNIRS], True)
ax4.plot(time_axis, acquisition.signal[:, 3])


ax5 = fig.add_subplot(gs[1, 3:6], sharex=ax1)
ax5.set_title('fNIRS 4')
ax5.plot(time_axis, acquisition.signal[:, 4])

ax6 = fig.add_subplot(gs[2, :3], sharex=ax1)
ax6.set_title('EEG 1')
acquisition.getBiosignalsSensors([acq.EEG], True)
ax6.plot(time_axis, acquisition.signal[:, 1])

ax7 = fig.add_subplot(gs[2, 3:6], sharex=ax1)
ax7.plot(time_axis, acquisition.signal[:, -1])

ax8 = fig.add_subplot(gs[3, :6], sharex=ax1)
ax8.set_title('EDA')
acquisition.getBiosignalsSensors([acq.EDA], True)
# eda = np.where(acquisition.sensors == acq.EDA)[0]
ax8.plot(time_axis, acquisition.signal[:, -1])


ax9 = fig.add_subplot(gs[3, 6:], sharex=ax1)
ax9.set_title('ECG')
# ecg = np.where(acquisition.sensors == acq.ECG)[0]
# ax2.plot(time_axis, biosignals[:, ecg])
acquisition.getBiosignalsSensors([acq.ECG], True)
ax9.plot(time_axis, acquisition.signal[:, -1])

ax10 = fig.add_subplot(gs[4, :6], sharex=ax1)
ax10.set_title('RESP')
# resp = np.where(acquisition.sensors == acq.RESP)[0]
# ax4.plot(time_axis, biosignals[:, resp])
acquisition.getBiosignalsSensors([acq.RESP], True)
ax10.plot(time_axis, acquisition.signal[:, -1])


ax11 = fig.add_subplot(gs[4, 6:], sharex=ax1)
ax11.set_title('Time to Answer')
ax11.scatter(time_answer[0], time_answer[1])

ax12 = fig.add_subplot(gs[5, :4], sharex=ax1)
ax12.set_title('X')
# acc = np.where(acquisition.sensors == acq.ACC)[0]
# ax10.plot(time_axis, biosignals[:, acc[0]])
acquisition.getBiosignalsSensors([acq.ACC], True)
ax12.plot(time_axis, acquisition.signal[:, 1])

ax13 = fig.add_subplot(gs[5, 4:8], sharex=ax1)
ax13.set_title('Y')
# ax11.plot(time_axis, biosignals[:, acc[1]])
ax13.plot(time_axis, acquisition.signal[:, 2])

ax14 = fig.add_subplot(gs[5, 8:], sharex=ax1)
ax14.set_title('Z')
# ax12.plot(time_axis, biosignals[:, acc[2]])
ax14.plot(time_axis, acquisition.signal[:, 3])

plt.draw()
plt.show(block=False)


# Comment these lines if you intend to continue the labelling process
with open(os.path.join(folder, 'custom_labels.txt'), 'w') as f:
    f.write('Timestamp,Label\n')

label = 'rest'
i = 0

while i < len(timestamps):
    timestamp = timestamps[i]
    ax1.set_xlim(timestamp-10, timestamp+10)
    
    img=mpimg.imread(os.path.join(folder_snapshots, f"{timestamp}.jpeg"))
    # ax3.imshow(img)
    picture.set_data(img)

    print(timestamp)
    plt.draw()
    plt.show(block = False)
    new_label = input("Type a label if something changed, else press Enter: ")
    if new_label == 'previous':
        with open(os.path.join(folder, 'custom_labels.txt'), 'r') as f:
            lines = f.readlines()
        print(lines[:-1])
        with open(os.path.join(folder, 'custom_labels.txt'), 'w') as f:
            f.writelines(lines[:-1])
        i -= 1
        continue
    if new_label is not '':
        label = new_label
    
    with open(os.path.join(folder, 'custom_labels.txt'), 'a') as f:
        f.write(f"{timestamp},{label}\n")
    i+=1
