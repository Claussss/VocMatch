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

#### Conclusion

Despite tuning the hyperparameters, the VAE was unable to cluster audios from the same project together, as evidenced by both the scatterplot and the spreadsheet. Embeddings generated from various audio representations, such as tempogram and chromagram, yield similar results. Increasing the quantity of vocal recordings and their corresponding accompaniments may enable the VAE to learn the patterns present in the data.

### Pitch Curve Approach

#### Intuition
This approach is based on the extraction of pitch curves from monophonic audio recordings and polyphonic accompaniments for the subsequent calculation of similarity.

#### Data

For the dataset I used vocals and the corresponding arrangments exported from my personal music projects.

I **normalized the volume** of every track,
so the arrangments and the vocals have the same volume range. 
I divided each vocal track into multiple **10-second segments**.

I manually selected a 60-second segment from each instrumental track which I believe best encapsulates the overall arrangement. I then proceeded to filter the segments by only taking frequencies between 20 and 200 Hz to extract the bassline, which is monophonic. By undertaking this, I can employ the same pitch extraction algorithm for both the vocal and arrangement.

#### Algorithm

1) I utilize CREPE to extract pitch curves from the vocal and bassline.
2) I utilize a Savitzky-Golay filter to smooth both of the pitch curves in order to focus the comparison more on the melody itself rather than small oscillations such as vibrato in vocal recordings.
3) By employing a sliding window technique, I am able to compute the Pearson correlation between the 10-second vocal pitch curve and each section of the 60-second bassline pitch curve.
4) Filter out sections wherein correlations are lower than the threshold.


![654](https://user-images.githubusercontent.com/48916506/210279502-0977a808-ee1c-4375-9056-f22fc692076b.png)


The final similarity is calculated by taking the average of the correlations between all sections.

#### Testing

To test the algorithm, I overlaid pitch curves and mixed the vocal with the original arrangement at the location with the greatest correlation.

![Screenshot_1](https://user-images.githubusercontent.com/48916506/210280107-ec4c94f8-0d59-4889-9a5d-fbb97e0ca287.png)

An example of audio generated from the algorithm's predictions is provided here:



https://user-images.githubusercontent.com/48916506/210280168-1eb9685a-00e9-473b-b829-d0e4cd2f9af7.mov


An example of the original song from which the vocal and arrangement were derived can be found here:


https://user-images.githubusercontent.com/48916506/210280218-40800adb-2789-4bce-83dc-6bcdc8d992ba.mov

It appears that the algorithm has positioned the vocal close to the original vocal's starting point. However, the algorithm also yields high similarity for vocals from other projects, as exemplified below.



https://user-images.githubusercontent.com/48916506/210280879-23d3344e-c795-4416-9e2d-29317dfdac78.mov


It is difficult say whether the melody of the second example is similar to the melody of the arrangement due to the algorithm's offset which displaces the vocals slightly. 


In order to test the algorithm automatically, I calculated correlations between a bassline and all the vocals, hypothesizing that vocals from the same project as the bassline would have a higher correlation.

An example of the spreadsheet with correlations for the "i_am_same_as_u" project is provided here.

![Screenshot_2](https://user-images.githubusercontent.com/48916506/210282314-8760757a-5be7-45d5-857b-359cd8eddb47.png)


#### Conclusion

The results yielded by the algorithm indicated that vocals from the same project as the bassline did not demonstrate a higher correlation, and vocals from the same project and sections did not exhibit a comparable Pearson correlation value in spite of vocalizing the same melody. 

It is possible that the bassline pitch curve may not accurately reflect the melody of an arrangement. Consequently, employing an algorithm that extracts pitch from non-monophonic audio to acquire a pitch curve from the original arrangement could potentially improve the results of the algorithm.
