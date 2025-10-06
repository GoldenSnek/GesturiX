import { StyleSheet, Text, View, TouchableOpacity, TextInput, ImageBackground, Alert } from 'react-native';
import React, { useState } from 'react';
import * as AuthSession from 'expo-auth-session';
import { supabase } from '../../src/supabaseClient';
import { router } from 'expo-router';

// ✅ Redirect URL for Expo Go (used for Google OAuth)
const redirectUrl = AuthSession.makeRedirectUri();

const SignUp: React.FC = () => {
  const [email, setEmail] = useState<string>('');
  const [password, setPassword] = useState<string>('');

  // 🔹 Email + Password Sign Up (Instant Account Creation)
  const handleSignUp = async (): Promise<void> => {
    const cleanEmail = email.trim().toLowerCase();
    const cleanPassword = password.trim();

    if (!cleanEmail || !cleanPassword) {
      Alert.alert('Error', 'Please fill in both fields');
      return;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(cleanEmail)) {
      Alert.alert('Error', 'Please enter a valid email address');
      return;
    }

    console.log('Checking for existing account:', cleanEmail);

    // 🔍 Check if email already exists in 'profiles' table
    const { data: existingUser, error: checkError } = await supabase
      .from('profiles')
      .select('email')
      .eq('email', cleanEmail)
      .maybeSingle();

    if (checkError) {
      console.error('Error checking existing email:', checkError);
      Alert.alert('Error', 'Something went wrong. Please try again later.');
      return;
    }

    if (existingUser) {
      Alert.alert('Error', 'An account with this email already exists.');
      return;
    }

    console.log('Attempting instant signup with:', cleanEmail);

    // 🚀 Instant account creation (no email verification)
    const { data, error } = await supabase.auth.signUp({
      email: cleanEmail,
      password: cleanPassword,
    });

    if (error) {
      console.log('Supabase signup error:', error);
      Alert.alert('Sign Up Failed', error.message);
      return;
    }

    // ✅ Directly log in user
    const { error: loginError } = await supabase.auth.signInWithPassword({
      email: cleanEmail,
      password: cleanPassword,
    });

    if (loginError) {
      console.error('Login after signup failed:', loginError);
      Alert.alert('Error', 'Account created, but auto-login failed. Please log in manually.');
      router.replace('/(stack)/Login');
      return;
    }

    Alert.alert('Success', 'Account created successfully!');
    router.replace('/(stack)/LandingPage');
// ✅ redirect to home or main screen
  };

  // 🔹 Google Sign Up
  const handleGoogleSignUp = async (): Promise<void> => {
    const { error } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: { redirectTo: redirectUrl },
    });

    if (error) {
      Alert.alert('Google Sign Up Failed', error.message);
    }
  };

  return (
    <ImageBackground
      source={require('../../assets/images/LoginSignUpBG.png')}
      className="flex-1 justify-center items-center p-8"
      resizeMode="cover"
    >
      <View className="absolute inset-0 bg-black opacity-40" />

      <View className="relative w-full max-w-sm p-8 rounded-3xl">
        <Text className="text-4xl font-bold text-black mb-10 text-center">Create Account</Text>

        <TextInput
          className="w-full border-2 border-accent rounded-lg p-4 mb-4 text-black text-lg font-bold bg-neutral"
          placeholder="Email"
          placeholderTextColor="#444444"
          value={email}
          onChangeText={setEmail}
          autoCapitalize="none"
        />
        <TextInput
          className="w-full border-2 border-accent rounded-lg p-4 mb-4 text-black text-lg font-bold bg-neutral"
          placeholder="Password"
          placeholderTextColor="#444444"
          secureTextEntry
          value={password}
          onChangeText={setPassword}
        />

        {/* Email Sign Up */}
        <TouchableOpacity
          onPress={handleSignUp}
          className="w-full bg-accent rounded-full py-4 items-center mt-4"
        >
          <Text className="text-white text-lg font-bold">Sign Up</Text>
        </TouchableOpacity>

        {/* Google Sign Up */}
        <TouchableOpacity
          onPress={handleGoogleSignUp}
          className="w-full bg-red-500 rounded-full py-4 items-center mt-4"
        >
          <Text className="text-white text-lg font-bold">Sign Up with Google</Text>
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
