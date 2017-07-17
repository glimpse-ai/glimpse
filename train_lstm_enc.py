"""
Train the LSTM encoder

Pix2Code Architecture

	DML (my designed Deep Markup Language)
		--> one-hot encoded token
			--> stack of 2 LSTM layers with 128 cells each
				--> output vector (combined with encoded image vector for more processing)

"""
from glimpse.encoders.lstm import LSTM

lstm = LSTM()
lstm.train()

