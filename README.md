# VocMatch
The concept of the project is to utilize chroma, tempo, and MFCC-based features to match existing arrangments to recorded vocal inputs.

The user interface of the final product will appear as follows:
1) The user vocalizes for 10 seconds
2) The algorithm searches the database with encoded fingerprints of arrangments
3) The algorithm produces a ranked list of the top K arrangements in order of similarity to the vocal.

The greater the similarity, the more agreeable the vocal and arrangement sound when mixed together.

This project is intended to facilitate the process of finding ready arrangements for songs for artists and to provide budding producers with an opportunity to promote their arrangements.

### VAE Approach

#### Intuition
This approach uses Variational Autoencoder to generate embeddings for mel-spectograms of vocals and arrangments. The cosine distance is utilized to calculate the similarity between vocal and arrangement embeddings.

#### Data

1) For the dataset I used vocals and the corresponding arrangments exported from my personal music projects. 
I **normalized the volume** of every track,
so the arrangments and the vocals have the same volume range. 
I divided each track into multiple **10-second segments**. In the end, 
I had approximately **2,000** 10-second pieces of vocals and arrangements in the **training** dataset, **400** in the validation dataset, and **400** in the testing dataset.

2) I applied the following audio augmentations randomly on the train dataset: 
**white noise, time stretch, and pitch scale**. (I used default functions from 
librosa.effects). As the result, I had about **6,000** elements in the **train** dataset.
3) I created normalized spectrograms from the audio dataset.
4) I converted the normalized log spectograms to png images with the size of 223x221.


Example of an **arrangement** audio and the corresponding mel-spectrogram from the **train** dataset:


https://user-images.githubusercontent.com/48916506/188779524-7d1ba81b-3647-4d74-89b4-cc309c96cf0a.mov

![baba_yaga_beat_section_1](https://user-images.githubusercontent.com/48916506/188780313-38da16d4-3beb-4ed4-8838-785ba30419eb.png)



Example of a vocal that is supposed to have a high similarity with the arrangement shown above because they are taken from the same project:



https://user-images.githubusercontent.com/48916506/188781071-2f0f0298-dd5b-460c-9651-d58e0d5bf31b.mov


![baba_yaga_vox1_section_1](https://user-images.githubusercontent.com/48916506/188781089-7286b865-7345-4715-a93c-b94376863894.png)



https://user-images.githubusercontent.com/48916506/188781099-701bd900-9a70-48b6-ac4a-138e4844fd45.mov


![baba_yaga_vox3_section_1](https://user-images.githubusercontent.com/48916506/188781106-d07d2b99-f95d-40db-8f57-3a0f1b7f04ed.png)



Example of vocals that are taken from a different project (their distance from the arrangement provided above in the embedding space is supposed to be bigger than the vocals provided above):



https://user-images.githubusercontent.com/48916506/188781443-43e57cf5-b8ba-424b-8928-5cec30cbbf17.mov


![be_afraid_my_enemy_vox1_section_1](https://user-images.githubusercontent.com/48916506/188781448-3b416199-a531-4be3-a34c-b4564a5a1ca0.png)



https://user-images.githubusercontent.com/48916506/188781466-1f7c4743-d7b3-4f98-97d5-c1f0afb5fb54.mov


![be_afraid_my_enemy_vox2_section_3](https://user-images.githubusercontent.com/48916506/188781480-41f90a7e-f7d4-42b9-a5e2-2aad760c2a08.png)


#### Training
I employed a Variational Autoencoder (VAE) with a ResNet18 backbone and the following hyperparameters for training:

| Hyperparameter    | Value |
|:------------------|:-----:|
| lr                | 1e-3  |
| epochs            |  100  |
| batch_size        |  16   |
| latent_space_size |  128  |

As the preprocessing, I converted the png images to **grayscale** and **resized** them to **(216,216)**.

#### Testing
I have conducted tests on the algorithm in two distinct ways.
First of all, I performed the following steps:
1) Selected 5 arrangements from the test dataset
2) Found the distances from every arrangement to every vocal
3) Sorted the results based on the distance, and saved them in a spreadsheet

I made the rows for vocals from the same project green, so, ideally, we would have all the 'green rows' on the top, which means they
have the smallest distances to a specific arrangement.

(you can find an example of a spreadsheet for one of the instrumentals in src)

![Screenshot_1](https://user-images.githubusercontent.com/48916506/188788033-d3b2789f-6c1a-4acc-a009-e765af27cc84.png)

Second of all, I used a dimensionality reduction algorithm on the embeddings, so that I could plot them as a scatter plot.
Crosses correpond to arrangements (instrumentals), and circles to vocals. Audio from the same project have the same size and color of markings.

![987](https://user-images.githubusercontent.com/48916506/210194479-1293945f-4659-41d4-a088-78f55a10e5f3.png)

### Ideas to try

Unfortunately, the current approach didn't work.
I observed that the model clusters audio pieces that were cut from 
the same audio track. So, the model clusters more based on acoustic features, and the properties of the sound itself,
which is understandable because I train on mel-spectrograms.

I think I should try using a visual representation of audio that captures some music features,
like chromogram. Or I should rely more on hand-crafted features like tempo or the key.

Also, a supervised approach can be explored. For example, training with pairs of a vocal and instrumental that 'sound good' together.

