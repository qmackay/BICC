import os
import re
import sys

dir = sys.argv[1]
inclusive = True

pattern = '.pdf'
regexObj = re.compile(pattern)
for root, dirs, files in os.walk(dir, topdown=False):
    for name in files:
        path = os.path.join(root, name)
        if bool(regexObj.search(path)) == bool(inclusive):
            os.remove(path)
    for name in dirs:
        path = os.path.join(root, name)
        if len(os.listdir(path)) == 0:
            os.rmdir(path)

pattern = '.svg'
regexObj = re.compile(pattern)
for root, dirs, files in os.walk(dir, topdown=False):
    for name in files:
        path = os.path.join(root, name)
        if bool(regexObj.search(path)) == bool(inclusive):
            os.remove(path)
    for name in dirs:
        path = os.path.join(root, name)
        if len(os.listdir(path)) == 0:
            os.rmdir(path)

pattern = '.png'
regexObj = re.compile(pattern)
for root, dirs, files in os.walk(dir, topdown=False):
    for name in files:
        path = os.path.join(root, name)
        if bool(regexObj.search(path)) == bool(inclusive):
            os.remove(path)
    for name in dirs:
        path = os.path.join(root, name)
        if len(os.listdir(path)) == 0:
            os.rmdir(path)

pattern = 'restart.'
regexObj = re.compile(pattern)
for root, dirs, files in os.walk(dir, topdown=False):
    for name in files:
        path = os.path.join(root, name)
        if bool(regexObj.search(path)) == bool(inclusive):
            os.remove(path)
    for name in dirs:
        path = os.path.join(root, name)
        if len(os.listdir(path)) == 0:
            os.rmdir(path)

pattern = '~'
regexObj = re.compile(pattern)
for root, dirs, files in os.walk(dir, topdown=False):
    for name in files:
        path = os.path.join(root, name)
        if bool(regexObj.search(path)) == bool(inclusive):
            os.remove(path)
    for name in dirs:
        path = os.path.join(root, name)
        if len(os.listdir(path)) == 0:
            os.rmdir(path)

pattern = '.bak'
regexObj = re.compile(pattern)
for root, dirs, files in os.walk(dir, topdown=False):
    for name in files:
        path = os.path.join(root, name)
        if bool(regexObj.search(path)) == bool(inclusive):
            os.remove(path)
    for name in dirs:
        path = os.path.join(root, name)
        if len(os.listdir(path)) == 0:
            os.rmdir(path)

pattern = 'output.txt'
regexObj = re.compile(pattern)
for root, dirs, files in os.walk(dir, topdown=False):
    for name in files:
        path = os.path.join(root, name)
        if bool(regexObj.search(path)) == bool(inclusive):
            os.remove(path)
    for name in dirs:
        path = os.path.join(root, name)
        if len(os.listdir(path)) == 0:
            os.rmdir(path)

pattern = '.pyc'
regexObj = re.compile(pattern)
for root, dirs, files in os.walk(dir, topdown=False):
    for name in files:
        path = os.path.join(root, name)
        if bool(regexObj.search(path)) == bool(inclusive):
            os.remove(path)
    for name in dirs:
        path = os.path.join(root, name)
        if len(os.listdir(path)) == 0:
            os.rmdir(path)
