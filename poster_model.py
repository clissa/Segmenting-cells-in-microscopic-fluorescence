n=2

inputs = Input((None, None, 3))

c0 = Conv2D(3, (3, 3), activation='relu',
            kernel_initializer='he_normal', padding='same')(inputs)

c1 = Conv2D(8*n, (7, 7), padding='same',  kernel_initializer='he_normal')(c0)
c1 = BatchNormalization()(c1)
c1 = Activation('elu')(c1)

c1 = Conv2D(8*n, (3, 3), padding='same', kernel_initializer='he_normal')(c1)

p1 = MaxPooling2D((2, 2))(c1)

##########################################
X_shortcut = Conv2D(16*n, (1, 1), padding='same', kernel_initializer='he_normal')(p1)
X_shortcut = BatchNormalization()(X_shortcut)

c2 = BatchNormalization()(p1)
c2 = Activation('elu')(c2)
c2 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c2)

c2 = BatchNormalization()(c2)
c2 = Activation('elu')(c2)
c2 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c2)

c2 = Add()([c2, X_shortcut])

p2 = MaxPooling2D((2, 2))(c2)

##########################################
X_shortcut = Conv2D(32*n, (1, 1), padding='same', kernel_initializer='he_normal')(p2)
X_shortcut = BatchNormalization()(X_shortcut)

c3 = BatchNormalization()(p2)
c3 = Activation('elu')(c3)
c3 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer='he_normal')(c3)

c3 = BatchNormalization()(c3)
c3 = Activation('elu')(c3)
c3 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer='he_normal')(c3)

c3 = Add()([c3, X_shortcut])

p3 = MaxPooling2D((2, 2))(c3)

###################Bridge#######################
# X_shortcut = Conv2D(64*n, (1, 1), padding='same',kernel_initializer='he_normal')(p3)
# X_shortcut = BatchNormalization()(X_shortcut)

X_shortcut = Conv2D(64*n, (1, 1), padding='same', kernel_initializer='he_normal')(p3)
X_shortcut = BatchNormalization()(X_shortcut)

c4 = BatchNormalization()(p3)
c4 = Activation('elu')(c4)
c4 = Conv2D(64*n, (5, 5), padding='same',kernel_initializer='he_normal')(c4)

c4 = BatchNormalization()(c4)
c4 = Activation('elu')(c4)
c4 = Conv2D(64*n, (5, 5), padding='same',kernel_initializer='he_normal')(c4)

# c41 = BatchNormalization()(c4)
# c41 = Activation('elu')(c41)
# c41 = Conv2D(64*n, (5, 5), padding='same',kernel_initializer='he_normal')(c41)

# c41 = BatchNormalization()(c41)
# c41 = Activation('elu')(c41)
# c41 = Conv2D(64*n, (5, 5), padding='same',kernel_initializer='he_normal')(c41)

c5 = BatchNormalization()(c4)
c5 = Activation('elu')(c5)
c5 = Conv2D(64*n, (5, 5), padding='same',kernel_initializer='he_normal')(c5)

c5 = BatchNormalization()(c5)
c5 = Activation('elu')(c5)
c5 = Conv2D(64*n, (5, 5), padding='same',kernel_initializer='he_normal')(c5)

c5 = Add()([c5, X_shortcut])

###################END BRIDGE#######################


u6 = Conv2DTranspose(32*n, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c3])

X_shortcut = Conv2D(32*n, (1, 1), padding='same', kernel_initializer='he_normal') (u6)
X_shortcut = BatchNormalization()(X_shortcut)

c6 = BatchNormalization()(u6)
c6 = Activation('elu')(c6)
c6 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer='he_normal')(c6)

c6 = BatchNormalization()(c6)
c6 = Activation('elu')(c6)
c6 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer='he_normal')(c6)

c6 = Add()([c6, X_shortcut])

################################################

u7 = Conv2DTranspose(16*n, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c2])

X_shortcut = Conv2D(16*n, (1, 1), padding='same', kernel_initializer='he_normal') (u7)
X_shortcut = BatchNormalization()(X_shortcut)

c7 = BatchNormalization()(u7)
c7 = Activation('elu')(c7)
c7 = Conv2D(16*n, (3, 3), padding='same',kernel_initializer='he_normal')(c7)

c7 = BatchNormalization()(c7)
c7 = Activation('elu')(c7)
c7 = Conv2D(16*n, (3, 3), padding='same',kernel_initializer='he_normal')(c7)

c7 = Add()([c7, X_shortcut])

u8 = Conv2DTranspose(8*n, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c1])

X_shortcut = Conv2D(8*n, (1, 1), padding='same', kernel_initializer='he_normal') (u8)
X_shortcut = BatchNormalization()(X_shortcut)

c8 = BatchNormalization()(u8)
c8 = Activation('elu')(c8)
c8 = Conv2D(8*n, (3, 3), padding='same',kernel_initializer='he_normal')(c8)

c8 = BatchNormalization()(c8)
c8 = Activation('elu')(c8)
c8 = Conv2D(8*n, (3, 3), padding='same',kernel_initializer='he_normal')(c8)

c8 = Add()([c8, X_shortcut])

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c8)

model = Model(inputs=[inputs], outputs=[outputs])

