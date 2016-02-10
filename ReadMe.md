REAL-TIME OBJECT DETECTION FOR "SMART" VEHICLES
-----------------------------------------------

Introduction
------------
The project presents an efficient method for object detection and demonstrates its use for real-time vision on-board vehicles. The work is closely based on Paul Viola’s face detection method [1], except that the system here was trained to detect objects related to traffic in order to assist the auto pilot on smart vehicles. The idea was adapted from D M Gavrila’s paper [2] REAL-TIME OBJECT DETECTION FOR "SMART" VEHICLES.

Objective
---------
The target is to create Haar feature-based cascade classifiers by training the system with an appropriate amount of traffic related information in order to detect vehicles, pedestrians and traffic signs in real-time. Thus, this system in addition to driving assistance/collision avoidance, could further find its application in tracking and recognition.

Haar-like Features
------------------
As described by Viola-Jones [1], Haar-like features are a simple and inexpensive image features based on intensity differences between rectangle-based regions that share similar shapes to the Haar wavelets and are defined as the difference of the sum of pixels of areas inside a rectangle and scale within the original image.