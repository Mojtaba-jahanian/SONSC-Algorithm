import numpy as np
from tensorflow import keras
from sonsc_algorithm import SONSC

def test_mnist():
    # بارگذاری MNIST
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    # پیش‌پردازش
    X = X_train.reshape(X_train.shape[0], -1)
    X = X.astype('float32') / 255.0
    
    # اجرای الگوریتم
    sonsc = SONSC(k_initial=2)
    sonsc.fit(X[:1000])  # استفاده از زیرمجموعه‌ای از داده‌ها برای سرعت بیشتر
    
    print("MNIST Results:")
    print(f"تعداد بهینه خوشه‌ها: {sonsc.k}")
    print(f"بهترین مقدار RCI: {sonsc.best_rci:.4f}")

def test_cifar10():
    # بارگذاری CIFAR-10
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    
    # پیش‌پردازش
    X = X_train.reshape(X_train.shape[0], -1)
    X = X.astype('float32') / 255.0
    
    # اجرای الگوریتم
    sonsc = SONSC(k_initial=2)
    sonsc.fit(X[:1000])  # استفاده از زیرمجموعه‌ای از داده‌ها برای سرعت بیشتر
    
    print("\nCIFAR-10 Results:")
    print(f"تعداد بهینه خوشه‌ها: {sonsc.k}")
    print(f"بهترین مقدار RCI: {sonsc.best_rci:.4f}")

if __name__ == "__main__":
    test_mnist()
    test_cifar10() 