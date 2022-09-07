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



Example of an **instrumental** audio and the corresponding mel-spectrogram from the **train** dataset:


https://user-images.githubusercontent.com/48916506/188779524-7d1ba81b-3647-4d74-89b4-cc309c96cf0a.mov

![baba_yaga_beat_section_1](https://user-images.githubusercontent.com/48916506/188780313-38da16d4-3beb-4ed4-8838-785ba30419eb.png)



Example of vocals that are supposed to sound good with the instrumental showed above because they are taken from the same project:



https://user-images.githubusercontent.com/48916506/188781071-2f0f0298-dd5b-460c-9651-d58e0d5bf31b.mov


![baba_yaga_vox1_section_1](https://user-images.githubusercontent.com/48916506/188781089-7286b865-7345-4715-a93c-b94376863894.png)



https://user-images.githubusercontent.com/48916506/188781099-701bd900-9a70-48b6-ac4a-138e4844fd45.mov


![baba_yaga_vox3_section_1](https://user-images.githubusercontent.com/48916506/188781106-d07d2b99-f95d-40db-8f57-3a0f1b7f04ed.png)



Example of vocals that are taken from a different project (their distance from the instrumental provided above in the embedding space is supposed to be bigger than the vocals provided above):



https://user-images.githubusercontent.com/48916506/188781443-43e57cf5-b8ba-424b-8928-5cec30cbbf17.mov


![be_afraid_my_enemy_vox1_section_1](https://user-images.githubusercontent.com/48916506/188781448-3b416199-a531-4be3-a34c-b4564a5a1ca0.png)



https://user-images.githubusercontent.com/48916506/188781466-1f7c4743-d7b3-4f98-97d5-c1f0afb5fb54.mov


![be_afraid_my_enemy_vox2_section_3](https://user-images.githubusercontent.com/48916506/188781480-41f90a7e-f7d4-42b9-a5e2-2aad760c2a08.png)


#### Training

#### Testing


### Ideas to try
