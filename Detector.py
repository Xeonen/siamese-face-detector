import os
from glob import glob
import torch
import torch.nn as nn
from torchvision import transforms

import pandas as pd
import numpy as np

from PIL import Image

from SiameseModels import SiameseNetworkTripletLoss

class ModelOps:
    """
    Class for managing the operations related to the model in the Siamese Face Detection system.
    """

    def __init__(self, model_loc=None, model=None) -> None:
        self.set_model(model_loc, model)
        self.set_device()
        self.set_transform()
        self.make_dirs()

    def set_model(self, model_loc, model) -> None:
        """
        Sets the model for face detection.

        Args:
            model_loc (str): The location of the pre-trained model.
            model (nn.Module): The pre-trained model. If None, a default model will be loaded.

        Returns:
            None
        """
        if model is None:
            self.model = SiameseNetworkTripletLoss()
            self.model.load_state_dict(torch.load(model_loc))
            self.model.eval()
        else:
            self.model = model

    def set_device(self) -> None:
        """
        Sets the device for model inference.

        Returns:
            None
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def set_transform(self) -> None:
        """
        Sets the image transformation pipeline.

        Returns:
            None
        """
        self.transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                             std=[0.229, 0.224, 0.225]),])
        self.inverse_transform = transforms.Compose([
                        transforms.Normalize(mean=[-0.485, -0.456, -0.406], 
                                             std=[1/0.229, 1/0.224, 1/0.225]),
                        transforms.ToPILImage()])

    def make_dirs(self) -> None:
        """
        Creates the necessary directories for storing data.

        Returns:
            None
        """
        self.bank_dir = os.path.join("data", "bank")
        self.process_dir = os.path.join("data", "process")
        os.makedirs(self.bank_dir, exist_ok=True)
        os.makedirs(self.process_dir, exist_ok=True)

class BankOps(ModelOps):
    """
    Class for performing operations on a bank of arrays.

    Args:
        model_loc (str): Location of the model.
        model: The model object.

    Attributes:
        process_dir (str): Directory containing the input images.
        bank_dir (str): Directory to store the bank of arrays.
        device: The device to use for computation.

    Methods:
        generate_arr: Generates and saves arrays for each folder in the process directory.
        remove_arr: Removes the array with the specified name from the bank.
        purge_arr: Removes all arrays from the bank.
        get_arr: Retrieves the array with the specified name from the bank.
        get_arr_dict: Retrieves all arrays from the bank as a dictionary.
    """

    def __init__(self, model_loc=None, model=None) -> None:
        super().__init__(model_loc, model)

    def generate_arr(self) -> None:
        """
        Generates and saves arrays for each folder in the process directory.
        """
        folder_list = glob(os.path.join(self.process_dir,"*"))

        for folder in folder_list:
            tmp_arr = []
            folder_name = os.path.split(folder)[-1]
            file_list = glob(os.path.join(folder, "*.png"))
            for file in file_list:
                img = Image.open(file)
                img = self.transform(img)
                img = img.to(self.device)
                img = img.unsqueeze(0)
                out = self.model(img)
                out = out.detach().cpu().numpy()
                tmp_arr.append(out)
            tmp_arr = np.array(tmp_arr)
            tmp_arr = np.mean(tmp_arr, axis=0)
            targat_loc = os.path.join(self.bank_dir, folder_name+ ".npy")
            if os.path.exists(targat_loc):
                os.remove(targat_loc)        
            np.save(os.path.join(self.bank_dir, folder_name), tmp_arr)

    def remove_arr(self, name) -> None:
        """
        Removes the array with the specified name from the bank.

        Args:
            name (str): The name of the array to remove.
        """
        path = os.path.join(self.bank_dir, name+".npy")
        if os.path.exists(path):
            os.remove(path)
            print("{name} removed")
        else:
            print("{name} not found")

    def remove_folder(self, name) -> None:
        """
        Removes the folder with the specified name from the process directory.

        Args:
            name (str): The name of the folder to remove.
        """
        path = os.path.join(self.process_dir, name)
        if os.path.exists(path):
            os.remove(path)

    def purge_folder(self) -> None:
        """
        Removes all folders from the process directory.
        """
        for name in os.listdir(self.process_dir):
            self.remove_folder(name)

    def purge_arr(self) -> None:
        """
        Removes all arrays from the bank.
        """
        for name in os.listdir(self.bank_dir):
            self.remove_arr(name.split(".")[0])

    def get_arr(self, name) -> np.array:
        """
        Retrieves the array with the specified name from the bank.

        Args:
            name (str): The name of the array to retrieve.

        Returns:
            np.array: The retrieved array.
        """
        path = os.path.join(self.bank_dir, name+".npy")
        if os.path.exists(path):
            return np.load(path)
        else:
            print("{name} not found")

    def get_arr_dict(self) -> dict:
        """
        Retrieves all arrays from the bank as a dictionary.

        Returns:
            dict: A dictionary containing the arrays, with the array names as keys.
        """
        arr_dict = {}
        for name in os.listdir(self.bank_dir):
            arr_dict[name.split(".")[0]] = self.get_arr(name.split(".")[0])
        return arr_dict


class Detector(ModelOps):
    def __init__(self, model_loc=None, model=None) -> None:
        """
        Initializes the Detector class.

        Args:
            model_loc (str): The location of the model file.
            model (nn.Module): The pre-trained model.

        Returns:
            None
        """
        super().__init__(model_loc)
        # self.set_model(model_loc)
        self.bank = BankOps(model=self.model)

    def detect(self, img, img_name) -> None:
        """
        Performs face detection on an input image.

        Args:
            img (PIL.Image.Image): The input image.
            img_name (str): The name of the image.

        Returns:
            dict: A dictionary containing the distances between the detected face and the reference faces.
        """
        img = self.transform(img)
        img = img.to(self.device)
        img = img.unsqueeze(0)
        out = self.model(img)
        out = out.detach().cpu()
        references = self.bank.get_arr_dict()
        distance_fn = nn.PairwiseDistance()
        distance_dict = {}
        for reference in references.keys():
            distance_dict[reference] = distance_fn(out, torch.tensor(references[reference])).item()

        distance_dict = dict(sorted(distance_dict.items(), key = lambda x: x[1], reverse = False))

        print("The most similar face is: ", list(distance_dict.keys())[0])

        return distance_dict


if __name__ == "__main__":
    img_id = "10144.png"
    img = Image.open(f"data/single_images/{img_id}")
    detector = Detector(model_loc="model_e_4_l_0.1467277131297434.pth")
    # Generates numpy arrays and saves them to the bank directory.
    # detector.bank.generate_arr() 
    # Removes all of the numpy arrays from the bank directory.
    # detector.bank.purge_arr()
    # Removes a specific numpy array from the bank directory.
    # detector.bank.remove_arr("10144")
    # Removes a specific folder from the process directory.
    # detector.bank.remove_folder("10144")
    # Removes all folders from the process directory.
    # detector.bank.purge_folder()
    detector_dict = detector.detect(img, img_id)
    for key in detector_dict.keys():
        print(key, detector_dict[key])
