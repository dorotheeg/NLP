def build_model(word_types, pos_types, outputs):
    # TODO: Write this function for part 3
    model = Sequential()
    model.add(Embedding(word_types, output_dim=32, input_length = 6))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(91, activation="softmax"))
    #model.add(...)
    model.compile(keras.optimizers.Adam(lr=0.01), loss="categorical_crossentropy")
    return model
