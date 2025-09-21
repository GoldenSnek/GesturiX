import { StyleSheet, Text, View, TouchableOpacity, TextInput, ImageBackground } from 'react-native';
import React from 'react';
import { router } from 'expo-router';

const SignUp = () => {
  const handleSignUp = () => {
    // Navigate back to Login
    router.replace('/(stack)/Login');
  };

  return (
    <ImageBackground
      source={require('../../assets/images/LoginSignUpBG.png')}
      className="flex-1 justify-center items-center p-8"
      resizeMode="cover"
    >
      {/* Semi-transparent overlay to make text more readable */}
      <View className="absolute inset-0 bg-black opacity-40" />
      
      {/* Sign Up Form Container */}
      <View className="relative w-full max-w-sm p-8 rounded-3xl">
        <Text className="text-4xl font-bold text-black mb-10 text-center">Create Account</Text>

        <TextInput 
          className="w-full border-2 border-accent rounded-lg p-4 mb-4 text-black text-lg font-bold bg-neutral"
          placeholder="Email"
          placeholderTextColor="#444444"
        />
        <TextInput 
          className="w-full border-2 border-accent rounded-lg p-4 mb-4 text-black text-lg font-bold bg-neutral"
          placeholder="Password"
          placeholderTextColor="#444444"
          secureTextEntry
        />

        <TouchableOpacity 
          onPress={handleSignUp}
          className="w-full bg-accent rounded-full py-4 items-center mt-4"
        >
          <Text className="text-white text-lg font-bold">Sign Up</Text>
        </TouchableOpacity>

        <View className="mt-6 flex-row items-center justify-center">
          <Text className="text-black">Already have an account? </Text>
          <TouchableOpacity onPress={() => router.replace('/(stack)/Login')}>
            <Text className="text-highlight font-bold">Log In</Text>
          </TouchableOpacity>
        </View>
      </View>
    </ImageBackground>
  );
};

export default SignUp;

const styles = StyleSheet.create({});