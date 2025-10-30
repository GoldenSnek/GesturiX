// app/(stack)/LandingPage.tsx
import { StyleSheet, Text, View, TouchableOpacity, ImageBackground, Image } from 'react-native';
import React from 'react';
import { router } from 'expo-router';

const LandingPage = () => {
  return (
    <ImageBackground
      source={require('../../assets/images/LandingPageBG.png')}
      className="flex-1 justify-between px-8 pt-4 pb-8"
      resizeMode="cover"
    >
      {/* overlay */}
      <View className="absolute inset-0 bg-black opacity-40" />

      <View className="z-10 items-center">
        <View className="mb-4 items-center">
          <Image
            source={require('../../assets/images/GesturiX-motto-color.png')}
            style={{ width: 320, height: 320 }}
            resizeMode="contain"
          />
        </View>
      </View>

      <View className="z-10 w-full mb-16 items-center">
        <TouchableOpacity
          onPress={() => router.push('/(stack)/SignUp')}
          className="w-full bg-accent rounded-full py-4 mb-4 items-center"
        >
          <Text className="text-white text-lg font-montserrat-bolditalic">Get Started</Text>
        </TouchableOpacity>

        <TouchableOpacity
          onPress={() => router.push('/(stack)/Login')}
          className="w-full border-2 border-accent rounded-full py-4 items-center"
        >
          <Text className="text-accent text-lg font-montserrat-bolditalic">Log In</Text>
        </TouchableOpacity>
      </View>
    </ImageBackground>
  );
};

export default LandingPage;

const styles = StyleSheet.create({});