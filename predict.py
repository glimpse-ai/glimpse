from glimpse.helpers import dataset
from glimpse.trainer import Trainer


if __name__ == '__main__':
  images, labels, label_lens = dataset.test()
  
  trainer = Trainer(feed_previous=True)

  image_batch = images[::4][:4]

  trainer.predict(image_batch)