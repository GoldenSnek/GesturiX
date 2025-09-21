import { StyleSheet, Text, View, TouchableOpacity, ImageBackground } from 'react-native';
import React from 'react';
import { router } from 'expo-router';

const LandingPage = () => {
  return (
    <ImageBackground
      source={require('../../assets/images/LandingPageBG.png')}
      className="flex-1 justify-between p-8"
      resizeMode="cover"
    >
      {/* Semi-transparent overlay to make text more readable */}
      <View className="absolute inset-0 bg-black opacity-40" />
      
      {/* Top Section: Title and Slogan */}
      <View className="z-10 items-center mt-16">
        <Text className="text-5xl font-bold text-accent mb-2">GesturiX</Text>
        <Text className="text-xl text-white/90 text-center">Your Pocket Sign Translator.</Text>
      </View>
      
      {/* Bottom Section: Buttons */}
      <View className="z-10 w-full mb-16 items-center">
        <TouchableOpacity 
          onPress={() => router.push('/(stack)/SignUp')}
          className="w-full bg-accent rounded-full py-4 mb-4 items-center"
        >
          <Text className="text-white text-lg font-bold">Get Started</Text>
        </TouchableOpacity>

        <TouchableOpacity 
          onPress={() => router.push('/(stack)/Login')}
          className="w-full border-2 border-accent rounded-full py-4 items-center"
        >
          <Text className="text-accent text-lg font-bold">Log In</Text>
        </TouchableOpacity>
      </View>
    </ImageBackground>
  );
};

export default LandingPage;

const styles = StyleSheet.create({});