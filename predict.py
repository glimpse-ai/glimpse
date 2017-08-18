from glimpse.trainer import Trainer


if __name__ == '__main__':
  trainer = Trainer(feed_previous=True)

  images = trainer.X_test
  image_batch = images[::4][:4]

  trainer.predict(image_batch)