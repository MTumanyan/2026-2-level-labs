"""
Programming 2026.

Seminar 13.

Polymorphism.
"""

# Animal
## Attributes
### eating_type (predator, herbivore)
### habitat
### moving_type
## Methods
### breathe
### eat
### sleep
## Magic Methods
### __str__
class Animal:
    def __init__(self, eating_type: str, habitat: str, moving_type: str) -> None:
        self.eating_type = eating_type
        self.habitat = habitat
        self.moving_type = moving_type
    def breathe(self) -> None:
        print("Breathing")
    def eat(self) -> None:
        print("Eating")
    def sleep(self) -> None:
        print("Sleeping")
    def __str__(self):
        return f"eating type: {self.eating_type}\nhabitat: {self.habitat}\nmoving_type: {self.moving_type}"

# Dog
## Attributes
### eating_type (predator)
### tamed (wild, domesticated)
### breed
### habitat
### moving_type
## Methods
### breathe
### eat
### sleep
## Magic Methods
### __str__

class Dog:
    def __init__(self, eating_type: str, tamed: str, breed: str, habitat: str) -> None:
        self.eating_type = eating_type
        self.tamed = tamed
        self.breed = breed
        self.habitat = habitat
    def breathe(self):
        return ("Breathing")
    def eat(self):
        return ("Eating")
    def sleep(self):
        return ("Sleeping")
# Cat
## Attributes
### eating_type (predator)
### habitat
### moving_type
### character
### fluffyness
## Methods
### breathe
### eat
### sleep
## Magic Methods
### __str__

# AnimalCollection
## Magic Methods
### __str__
### __len__
### __getitem__
## Advanced
### __setitem__
### __next__
### __iter__

# One code to process different types

# bubble = Cat(habitat='flat', character='playful', fluffiness='very')

# animals = [pug, bubble]
# for animal in animals:
#     animal.eat('meat')


def main() -> None:
    """
    Launch listing.
    """


if __name__ == "__main__":
    main()
