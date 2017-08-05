import numpy as np
from glimpse import model
import glimpse.helpers.definitions as definitions

model_path = definitions.model_path

m = model.Model(model_path, feed_previous=True)

#@Ben: replace this with actual data
images = np.random.randn(4,1250,640,3)

predictions = m.batch_predict(images)
