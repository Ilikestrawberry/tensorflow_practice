import tensorflow_datasets as tfds

dataset, ds_info = tfds.load(name='mnist',
                             shuffle_files=True,
                             with_info=True)

print(ds_info)  ## dataset 정보
print(ds_info.features)
print(ds_info.splits)

# %%
dataset = tfds.load(name='mnist', shuffle_files=True)

print(type(dataset))
print(dataset.keys(), '\n')
print(dataset.values())

train_ds = dataset['train'].batch(32)  ## dataset에서 train을 32개씩
test_ds = dataset['test']

# print(type(test_ds))
EPOCHS = 10

for epoch in range(EPOCHS):
    for data in train_ds:
        images = data['image']
        labels = data['label']

for tmp in train_ds:
    print(type(tmp))
    print(tmp.keys())
    images = tmp['image']
    labels = tmp['label']

    print(images.shape)
    print(labels.shape)
    break

# %%
dataset = tfds.load(name='mnist', shuffle_files=True, as_supervised=True) ## as_supervised -> 튜플형식

train_ds = dataset['train'].batch(32)
test_ds = dataset['test']

for tmp in train_ds:
    # print(type(tmp))
    images = tmp[0]
    labels = tmp[1]

    print(images.shape)
    print(labels.shape)
    break

# %%
train_ds, test_ds = tfds.load(name='mnist',
                        shuffle_files=True,
                        as_supervised=True,
                        split=['train', 'test'])


# %%
(train_ds, validation_ds, test_ds), ds_info = tfds.load(name='patch_camelyon',
                                                        shuffle_files=True,
                                                        as_supervised=True,
                                                        with_info=True,
                                                         split=['train', 'validation', 'test'])

# print(ds_info.features, '\n')
# print(ds_info.splits)

import matplotlib.pyplot as plt

train_ds = train_ds.batch(16)

train_ds_iter = iter(train_ds)
images, labels = next(train_ds_iter)
images = images.numpy()
labels = labels.numpy()

fig, axes = plt.subplots(4, 4, figsize=(7, 7))

for ax_idx, ax in enumerate(axes.flat):
    ax.imshow(images[ax_idx, ...])
    ax.set_title(labels[ax_idx], fontsize=10)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

