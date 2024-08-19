import numpy as np
from torch.utils.data import Dataset
import os
import csv
from PIL import Image
import customTransform
import torch


class CustomDatasetFCM(Dataset):
    def __init__(self, root_dir, split, Rescale, RandomCrop, Mirror, **kwargs):
        """custom dataset for FCM

        Args:
            root_dir (string): root directory filepath e.g. (dir1/dir2)
            split (string): filepath for split data e.g. (train.csv)
            Rescale (_type_): _description_
            RandomCrop (_type_): _description_
            Mirror (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.root_dir = root_dir
        self.split = split
        self.Rescale = Rescale
        self.RandomCrop = RandomCrop
        self.Mirror = Mirror
        self.hidden_state_dim = 150
        self.num_elements = 0
        self.tweet_ids = []
        self.labels = []
        self.tweets = None
        self.img_texts = None

        self.get_num_elements(f"{root_dir}/{split}")
        self.set_tweet_ids(f"{root_dir}/{split}")
        self.set_labels(f"{root_dir}/{split}")

        # initially set all tweets and img text to zeros
        self.tweets = np.zeros(
            (self.num_elements, self.hidden_state_dim), dtype=np.float32
        )
        self.img_texts = np.zeros(
            (self.num_elements, self.hidden_state_dim), dtype=np.float32
        )

        if 'embed_dir' in kwargs:
            self.set_tweet_text(f"{root_dir}/{kwargs['embed_dir']}/tweet_txt.txt")
            self.set_image_texts(f"{root_dir}/{kwargs['embed_dir']}/image_txt.txt")

    # get number of elements or rows of split
    def get_num_elements(self, split_filepath):
        with open(split_filepath, "r", encoding="utf-8") as file:
            csv_reader = csv.reader(file)
            data = [row for row in csv_reader]
            self.num_elements = len(data)

        # assumes that split (csv) has a header so we have to decrement 1
        if self.num_elements > 0:
            self.num_elements -= 1

    # set tweet ids from split
    def set_tweet_ids(self, split_filepath):
        with open(split_filepath, "r", encoding="utf-8") as file:
            index = 0
            csv_reader = csv.reader(file)
            data = [row for row in csv_reader]
            #  remove header
            if len(data) > 0:
                data = data[1:]

            for row in data:
                self.tweet_ids.append(row[0])

    # set labels from split
    def set_labels(self, split_filepath):
        with open(split_filepath, "r", encoding="utf-8") as file:
            csv_reader = csv.reader(file)
            data = [row for row in csv_reader]
            #  remove header
            if len(data) > 0:
                data = data[1:]

            for row in data:
                self.labels.append(row[-2])
    
    def set_tweet_text(self, filepath):
        try:
            for i,line in enumerate(open(filepath)):
                data = line.strip().split(',')
                for c in range(self.hidden_state_dim): 
                    self.tweets[i,c] = float(data[c+1])
        except Exception as e:
            print('error', e)

    def set_image_texts(self, filepath):
        try:
            for i,line in enumerate(open(filepath)):
                data = line.strip().split(',')
                tweet_id = data[0]

                if tweet_id in self.tweet_ids:
                    arr = np.array(list(map(float, data[1:])))
                    index = self.tweet_ids.index(tweet_id)
                    self.img_texts[index, :] = arr
        except Exception as e:
            print('error', e)


    def __len__(self):
        return len(self.tweet_ids)

    def __getitem__(self, index):
        img_name = f"{self.root_dir}/images_resized/{self.tweet_ids[index]}.jpg"
        out_img = np.zeros((3, 299, 299), dtype=np.float32)  # tweet image
        out_text = np.zeros(self.hidden_state_dim)  # tweet text
        out_img_text = np.zeros(self.hidden_state_dim)  # tweet image text

        # for handling image only
        try:
            image = Image.open(img_name)
            width, height = image.size

            if self.RandomCrop >= width or self.RandomCrop >= height:
                image = image.resize(
                    (int(width * 1.5), int(height * 1.5)), Image.LANCZOS
                )

            if self.Rescale != 0:
                image = customTransform.rescale(image, self.Rescale)

            if self.RandomCrop != 0:
                image = customTransform.random_crop(image, self.RandomCrop)

            if self.Mirror:
                image = customTransform.mirror(image)

            image = customTransform.preprocess_image_to_np_arr(image)

            out_img = image
        except:
            out_img = np.zeros((3, 299, 299), dtype=np.float32)

        # for handling tweet text only
        try:
            out_text = torch.from_numpy(np.array(self.tweets[index]).copy())
        except:
            out_text = torch.from_numpy(np.array(self.tweets[index]).copy())

        # for handling image text only
        try:
            out_img_text = torch.from_numpy(np.array(self.img_texts[index]).copy())
        except:
            out_img_text = torch.from_numpy(np.array(self.img_texts[index]).copy())

        label = torch.from_numpy(np.array([int(self.labels[index])]))
        label = label.type(torch.LongTensor)

        return torch.from_numpy(out_img.copy()), out_img_text, out_text, label
    


class CustomDatasetLFCM(Dataset):
    def __init__(self, root_dir, split, Rescale, RandomCrop, Mirror, **kwargs):
        """custom dataset for FCM

        Args:
            root_dir (string): root directory filepath e.g. (dir1/dir2)
            split (string): filepath for split data e.g. (train.csv)
            Rescale (_type_): _description_
            RandomCrop (_type_): _description_
            Mirror (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.root_dir = root_dir
        self.split = split
        self.Rescale = Rescale
        self.RandomCrop = RandomCrop
        self.Mirror = Mirror
        self.hidden_state_dim = 150
        self.num_elements = 0
        self.tweet_ids = []
        self.labels = []
        self.tweets = None
        self.img_texts = None
        self.comments = None
    

        self.get_num_elements(f"{root_dir}/{split}")
        self.set_tweet_ids(f"{root_dir}/{split}")
        self.set_labels(f"{root_dir}/{split}")

        # initially set all tweets and img text, comments to zeros
        self.tweets = np.zeros(
            (self.num_elements, self.hidden_state_dim), dtype=np.float32
        )
        self.img_texts = np.zeros(
            (self.num_elements, self.hidden_state_dim), dtype=np.float32
        )
        self.comments = np.zeros(
            (self.num_elements, self.hidden_state_dim), dtype=np.float32
        )

        self.set_tweet_text(f"{root_dir}/{kwargs['embed_dir']}/tweet_txt.txt")

        if 'embed_dir' in kwargs:
            self.set_image_texts(f"{root_dir}/{kwargs['embed_dir']}/image_txt.txt")
            self.set_comments(f"{root_dir}/{kwargs['embed_dir']}/comments.txt")

    # get number of elements or rows of split
    def get_num_elements(self, split_filepath):
        with open(split_filepath, "r", encoding="utf-8") as file:
            csv_reader = csv.reader(file)
            data = [row for row in csv_reader]
            self.num_elements = len(data)

        # assumes that split (csv) has a header so we have to decrement 1
        if self.num_elements > 0:
            self.num_elements -= 1

    # set tweet ids from split
    def set_tweet_ids(self, split_filepath):
        with open(split_filepath, "r", encoding="utf-8") as file:
            index = 0
            csv_reader = csv.reader(file)
            data = [row for row in csv_reader]
            #  remove header
            if len(data) > 0:
                data = data[1:]

            for row in data:
                self.tweet_ids.append(row[0])

    # set labels from split
    def set_labels(self, split_filepath):
        with open(split_filepath, "r", encoding="utf-8") as file:
            csv_reader = csv.reader(file)
            data = [row for row in csv_reader]
            #  remove header
            if len(data) > 0:
                data = data[1:]

            for row in data:
                self.labels.append(row[-2])
    
    def set_tweet_text(self, filepath):
        try:
            for i,line in enumerate(open(filepath)):
                data = line.strip().split(',')
                for c in range(self.hidden_state_dim): 
                    self.tweets[i,c] = float(data[c+1])
        except Exception as e:
            print('error', e)

    def set_image_texts(self, filepath):
        try:
            for i,line in enumerate(open(filepath)):
                data = line.strip().split(',')
                tweet_id = data[0]

                if tweet_id in self.tweet_ids:
                    arr = np.array(list(map(float, data[1:])))
                    index = self.tweet_ids.index(tweet_id)
                    self.img_texts[index, :] = arr
        except Exception as e:
            print('error', e)

    def set_comments(self, filepath):
        try:
            for i,line in enumerate(open(filepath)):
                data = line.strip().split(',')
                tweet_id = data[0]

                if tweet_id in self.tweet_ids:
                    arr = np.array(list(map(float, data[1:])))
                    index = self.tweet_ids.index(tweet_id)
                    self.comments[index, :] = arr
        except Exception as e:
            print('error', e)


    def __len__(self):
        return len(self.tweet_ids)

    def __getitem__(self, index):
        img_name = f"{self.root_dir}/images_resized/{self.tweet_ids[index]}.jpg"
        out_img = np.zeros((3, 299, 299), dtype=np.float32)  # tweet image
        out_text = np.zeros(self.hidden_state_dim)  # tweet text
        out_img_text = np.zeros(self.hidden_state_dim)  # tweet image text
        comment_text =  np.zeros(self.hidden_state_dim) # comment 

        # for handling image only
        try:
            image = Image.open(img_name)
            width, height = image.size

            if self.RandomCrop >= width or self.RandomCrop >= height:
                image = image.resize(
                    (int(width * 1.5), int(height * 1.5)), Image.LANCZOS
                )

            if self.Rescale != 0:
                image = customTransform.rescale(image, self.Rescale)

            if self.RandomCrop != 0:
                image = customTransform.random_crop(image, self.RandomCrop)

            if self.Mirror:
                image = customTransform.mirror(image)

            image = customTransform.preprocess_image_to_np_arr(image)

            out_img = image
        except:
            out_img = np.zeros((3, 299, 299), dtype=np.float32)

        # for handling tweet text only
        try:
            out_text = torch.from_numpy(np.array(self.tweets[index]).copy())
        except:
            out_text = torch.from_numpy(np.array(self.tweets[index]).copy())

        # for handling image text only
        try:
            out_img_text = torch.from_numpy(np.array(self.img_texts[index]).copy())
        except:
            out_img_text = torch.from_numpy(np.array(self.img_texts[index]).copy())

        try:
            comment_text = torch.from_numpy(np.array(self.comments[index]).copy())
        except:
            comment_text = torch.from_numpy(np.array(self.comments[index]).copy())
        

        label = torch.from_numpy(np.array([int(self.labels[index])]))
        label = label.type(torch.LongTensor)

        return torch.from_numpy(out_img.copy()), out_img_text, out_text, comment_text, label
