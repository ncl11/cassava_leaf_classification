__author__ = "Nathaniel Lanier"

"""
investigate_misclassification class which can be used to view misclassified images
"""

from matplotlib import pyplot as plt
import numpy as np

class investigate_misclass:
  """Creates an instance of the investigate_misclassification class from a pretrained model and a validation data generator"""
  def __init__(self, model, data_generator):
    self.model = model
    self.data_generator = data_generator
    self.misclassified_master = []
    self.classes = {
        "0":"Cassava Bacterial Blight (CBB)",
        "1":"Cassava Brown Streak Disease (CBSD)",
        "2":"Cassava Green Mottle (CGM)",
        "3":"Cassava Mosaic Disease (CMD)",
        "4":"Healthy"
        }

  def find_misclass(self, n_img):
    """Finds the first n_img misclassified images"""
    total_count = 0
    misclass_count = 0
    for img in self.data_generator:
      total_count += 1
      pred_prob = self.model.predict(img[0])
      pred_class = np.argmax(pred_prob)
      if pred_class != np.array(img[1])[0]:
        self.misclassified_master.append(img + (pred_class, ))
        misclass_count += 1
      if misclass_count == n_img:
        break
    print("Done")
    print(f"{n_img} misclassified images saved out of {total_count} images scanned")
  
  def print_misclass(self, start, stop, true=None, pred=None):
    """method to print misclassified images from index start to stop, indexing starts at 0, 
      optional true and/or pred parameters for filtering on class"""
    for i in range(start, stop):
      img = self.misclassified_master[i]
      true_class = np.array(img[1])[0]
      pred_class = img[2]
      if true:
        if true_class != true:
          continue
      if pred:
        if pred_class != pred:
          continue
      true_class = self.classes[str(true_class)]
      pred_class = self.classes[str(pred_class)]
      plt.imshow(img[0][0]/255, interpolation='nearest')
      plt.title(f'True Class: {true_class} - Pred Class: {pred_class}')
      plt.show()