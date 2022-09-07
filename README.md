# VocMatch
The idea of the project is to automatically match existing instruments to recorded vocal inputs that sound harmonic within rhythmic and textural analysis.


The final user interface would look like:
1) User sings for 20 seconds
2) The program searches the database with encoded fingerprints of instrumentals
3) The program outputs a sorted list of top K instrumentals that 'sound nice' with the recorded vocal 


### VAE Approach

#### Intuition
The current approach relies on using Variational Autoencoder. Namely, I use the Encoder to produce 
embeddings of mel-spectograms of instrumentals and vocals, and I try to find the cosine distance between the embeddings.
The smaller the distance, the more similar are the audio fragments. Thus, if we try to find the distance
between the embedding of a vocal and an instrumental, the smaller the distance, the "better" they sound together. 

#### Data

1) For the dataset I used vocal tracks and the corresponding instrumental tracks 
exported from my personal music projects. 
I **normalized the volume** of every track,
so the instrumentals and the vocals have the same volume range. 
I split every track on multiple **20 second sections**. In the end, 
I had about **1k** 20-second pieces of vocals and instrumentals in **train** dataset, 
**200** in the valid, and **200** in the test dataset.

2) I applied the following audio augmentations randomly on the train dataset: 
**white noise, time stretch, and pitch scale**. (I used default functions from 
librosa.effects). As the result, I had about **3k** images in the **train** dataset.
3) I created normalized spectrograms from the audio dataset using the following code:

``
stft = librosa.stft(signal, n_fft=FRAME_SIZE,  hop_length=HOP_LENGTH)[:-1]
``
<br />
``spectogram = np.abs(stft)``
<br />
``log_spectogram = librosa.amplitude_to_db(spectogram)``
<br />
``normalized_log_spectogram = (log_spectogram - log_spectogram.min()) / (log_spectogram.max() - log_spectogram.min())``

4) I converted the normalized log spectograms to png images with the size of 223x221 using the following code:


``
fig = plt.figure(figsize=(0.72, 0.72))
   ``
   <br />
   ``ax = fig.add_subplot(111)``
   <br />
   ``ax.axes.get_xaxis().set_visible(False)``
   <br />
   ``ax.axes.get_yaxis().set_visible(False)``
    <br />
    ``ax.set_frame_on(False)``
    <br />
    ``librosa.display.specshow(spectogram)``
    <br />
    ``plt.savefig(save_path, dpi=400, bbox_inches='tight', pad_inches=0)``




https://user-images.githubusercontent.com/48916506/188779524-7d1ba81b-3647-4d74-89b4-cc309c96cf0a.mov

![baba_yaga_beat_section_1](https://user-images.githubusercontent.com/48916506/188780313-38da16d4-3beb-4ed4-8838-785ba30419eb.png)

#### Training

#### Testing


### Ideas to try
