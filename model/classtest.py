# -*- coding: utf-8 -*-
# @Time : 2024/9/23 20:26
# @Author : CSR
# @File : classtest.py

from pydantic import BaseModel
from typing import List

class User(BaseModel):
    name: str
    age: int
    email: str
    friends: List[str] = []

class CreateUser(User):

    def __init__(self, name: str, age: int, email: str, friends: List[str] = []):
        super().__init__(name=name, age=age, email=email, friends=friends)


    def process(self):
        # print(self.friends)
        print(self.name)
        return self

if __name__ == '__main__':
    test = CreateUser(name="John Doe", age=30, email="john@example.com", friends = ["Alice", "Bob"])
    test.process()

