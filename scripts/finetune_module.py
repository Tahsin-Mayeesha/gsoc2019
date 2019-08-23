import tensorflow.compat.v2 as tf
import pickle

module = tf.saved_model.load("language_model")
 
print("Getting IMDB Reviews")
with open("data/imdb_reviews.pkl",'rb') as f:
    imdb_reviews = pickle.load(f)
    
print(len(imdb_reviews),"reviews")

imdb_dataset = tf.data.Dataset.from_tensor_slices(imdb_reviews).batch(16,drop_remainder=True)


train_losses = []
step = 0
for epoch in range(1):
    
    for batch in imdb_dataset:
        train_loss = module.train(batch)
        print("Step ",step," Train loss: ",train_loss.numpy())
        if step%100==0:
            tf.saved_model.save(module,"finetuned_language_model")
        train_losses.append(train_loss)
        step +=1
    # saving model per epoch
    
with open("finetune_losses.pkl",'wb') as f:
    pickle.dump(train_losses,f)
