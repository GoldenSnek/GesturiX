import { StyleSheet, Text, View, TouchableOpacity, ImageBackground, Image } from 'react-native'; // 👈 Added Image
import React from 'react';
import { router } from 'expo-router';

const LandingPage = () => {
  return (
    <ImageBackground
      source={require('../../assets/images/LandingPageBG.png')}
      // 🎯 MODIFIED: Changed p-8 to specific padding: reduced top padding (pt-4) 
      // to pull the content up, kept horizontal (px-8) and bottom (pb-8) padding.
      className="flex-1 justify-between px-8 pt-4 pb-8" 
      resizeMode="cover"
    >
      {/* Semi-transparent overlay to make text more readable */}
      <View className="absolute inset-0 bg-black opacity-40" />
      
      {/* 🖼️ Top Section: Image Replacement */}
      {/* 🎯 MODIFIED: Removed mt-16 to allow the section to move up to the new pt-4 padding */}
      <View className="z-10 items-center">
        
        {/* 1. Image Container */}
        <View className="mb-4 items-center">
          <Image
            // 👈 PLACEHOLDER PATH: Change this to your actual file path
            source={require('../../assets/images/GesturiX-motto-color.png')}
            // 🎯 ADDED: Use style for guaranteed sizing if className fails, or use NativeWind classes
            style={{ width: 320, height: 320 }} 
            resizeMode="contain" 
          />
        </View>
        
        {/* ❌ DELETED: The two original text lines are completely removed */}
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