# -----------------------------------------------------------
# Splits all the audios in the specified folder to a smaller sections of the same length
# specified by DEFAULT_DURATION_SECONDS.
# -----------------------------------------------------------



import os
from tqdm import tqdm
from pydub import AudioSegment, effects

# Every section is 20 sec
DEFAULT_DURATION_SECONDS = 20
path_to_result = r'D:\Education\Projects\FiZam\data\sections'
path_to_files = r'D:\Education\Projects\FiZam\data\full_beats'

if __name__ == '__main__':
    file_names = os.listdir(path_to_files)
    for file_name in tqdm(file_names):
        full_path_to_file = os.path.join(path_to_files,file_name)
        if 'mp3' in file_name:
            file = AudioSegment.from_mp3(full_path_to_file)
        elif 'wav' in file_name:
            file = AudioSegment.from_wav(full_path_to_file)
        else:
            print(f'{file_name} does not have a proper extension.')
            continue

        # Normalize the volume
        file = effects.normalize(file)
        sections_num = int(file.duration_seconds//DEFAULT_DURATION_SECONDS)
        for i in range(sections_num):
            section = file[i*DEFAULT_DURATION_SECONDS*1000:(i+1)*DEFAULT_DURATION_SECONDS*1000]
            # Skip sections with silence
            if section.max > 0:
                new_file_name = f'{file_name[:-4]}_section_{i+1}.wav'
                section.export(os.path.join(path_to_result, new_file_name), format='wav')
            else:
                print('Silence was skipped..')
