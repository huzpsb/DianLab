# Report

Implemented the 2 TODO blocks and evaluate on MNIST, the accuracy is around 95%.

All the parts that need to be implemented are finished.

## Advantages
* Algorithm : knn

  It's easy to write and maintain.Also, not hard to implement in another language or another place.
* Accuracy : 95%

  Useful even in a production environment.
* Train : 13μs / point

  Quick to train means it can be used for customized models of whatever.

## Disadvantages
* Test: 90ms / point

    Making it nearly impossible to serve on a server.
* No pre-compiled models

    Sometimes the raw dataset can't be published, and it won't work.

## Author
* gzh(independently)

## Acknowledgements
Thanks to:

https://github.com/caio-ggomes/CT-213 (Lab-9, For providing an example to deserialize the data)

https://blog.csdn.net/weixin_37830912/article/details/100159324 (For providing the way to unzip .gz in python)
