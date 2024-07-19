# Tips

## Additional Libraries: torch_scatter torch_sparse torch_cluster

[https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#additional-libraries](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#additional-libraries)

- Get PyTorch and CUDA version on your computer

```python
import torch

if __name__ == "__main__":
    print(torch.__version__)
    print(torch.cuda_version)

# 2.0.0
# 11.8
```

- Install the specific PyTorch and CUDA versions respectively

```cmd
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## Python: self & cls

`cls`是类方法中的参数, 用于指代类自身; 而`self`是实例方法中的参数, 用于指代实例对象

- 类方法 `class method`: 用于操作类层面的数据和行为
- 实例方法 `instance method`: 用于操作实例层面的数据和行为

```python
class Car(object):
    def __init__(self, brand, color):
        self.brand = brand
        self.color = color

    # instance method
    def show_info(self):
        print(f"This is a {self.color} {self.brand} car.")

    @classmethod
    def show_info_class(cls, brand, color):
        car = cls(brand, color)  # 创建实例对象
        car.show_info()  # 调用实例方法

# 1. 类方法
Car.show_info_class("BMW", "blue")

# 2. 实例方法
car = Car("BMW", "blue")
car.show_info()
```

## 装饰器: @classmethod, @staticmethod, @property

### @classmethod

在使用的时候, 会将类本身作为第一个参数 `cls` 传递给类方法

```python
class Web(object):

    name = "Python_Web"

    def __init__(self):
        self.desc = "实例属性, 不共享"

    def norm_method(self):
        """普通方法"""
        print('普通方法被调用!')

    @staticmethod
    def foo_staticmethod():
        print('静态方法被调用!')

    @classmethod
    def foo_classmethod_other(cls):
        print('另外一个类方法被调用!')

    # 类方法, 第一个参数为cls, 代表类本身
    @classmethod
    def foo_classmethod(cls):
        # 1. 调用静态变量
        print(cls.name)
        print(Web.name)

        # 2. 调用其他类方法
        cls.foo_classmethod_other()

        # 3. 调用静态方法
        cls.foo_staticmethod()

        # 4. 要调用实例属性, 必须使用cls实例化一个对象, 然后再去引用
        print(cls().desc)

        # 5. 要调用普通方法, 必须使用cls实例化一个对象, 然后再去引用
        cls().norm_method()


if __name__ == '__main__':
    # 使用类名去调用类方法
    Web.foo_classmethod()
```

### @staticmethid

```python
class Web(object):
    # 静态变量（类变量）
    name = "Python_Web"

    def __init__(self):
        self.desc = "实例属性, 不共享"

    def norm_method(self):
        """普通方法"""
        print('普通方法被调用!')

    @classmethod
    def foo_classmethod_other(cls):
        print('类方法被调用!')

    @staticmethod
    def foo_staticmethod_other():
        print('另外一个静态方法被调用!')

    @staticmethod
    def foo_staticmethod():
        """静态方法"""
        # 1. 调用其他静态方法
        print(Web.foo_staticmethod_other()) 

        # 2. 调用类方法
        print(Web.foo_classmethod_other())

        # 3. 引用静态变量
        print(Web.name)

        # 4. 调用普通方法, 访问实例属性
        # 必须通过实例对象去引用, 不能直接使用类名去访问
        instance = Web()
        print(instance.desc)
        instance.norm_method()


if __name__ == "__name__":
    # 推荐: 直接使用类名+方法名调用
    Web.foo_staticmethod()
    # 不推荐: 实例化一个类对象, 通过这个对象去调用静态方法
    instance = Web()
    instance.foo_staticmethod()  
```

### @property

负责把一个方法变成属性调用

```python
class Student(object):

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        if not isinstance(value, int):
            raise ValueError('score must be an integer!')
        if value < 0 or value > 100:
            raise ValueError('score must between 0 ~ 100!')
        self._score = value
```

```shell
>>> s = Student()
>>> s.score = 60 # OK, 实际转化为s.set_score(60)
>>> s.score # OK, 实际转化为s.get_score()
60
>>> s.score = 9999
Traceback (most recent call last):
  ...
ValueError: score must between 0 ~ 100!
```

## 打印运行过程信息

```python
print(f"\n{'-' * 20}    Start    {'-' * 20}\n")
print(f"\n{'-' * 20}    End    {'-' * 20}\n")
```

## 常用函数

- `isinstance(name, type)`: Return whether an object is an instance of a class or of a subclass thereof.
- `__getitem__` & `__setitem__`: 创建自定义的字典类, 并对其行为进行个性化定制
- `getattr()` & `setattr()`: Get or set a named attribute from an object
- `vars()`: 函数返回对象object的属性和属性值的字典对象, Without arguments, equivalent to `locals()`, With an argument, equivalent to `object.__dict__`

## 断言: assert

用于判断一个表达式, 在表达式条件为 `false` 的时候触发异常, 直接返回错误, 而不必等待程序运行后出现崩溃的情况

```python
assert expression [, arguments]

# 等价于
if not expression:
    raise AssertionError(arguments)

# Example:
assert self.reusable == False, "self.reusable == True Unsupported currently!"
```
