import { StyleSheet, Text, View, TouchableOpacity, TextInput, ImageBackground, Alert } from 'react-native';
import React, { useState } from 'react';
import { supabase } from '../../src/supabaseClient';
import { router } from 'expo-router';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  // 🔹 Handle Login
  const handleLogin = async () => {
    const cleanEmail = email.trim().toLowerCase();
    const cleanPassword = password.trim();

    if (!cleanEmail || !cleanPassword) {
      Alert.alert('Error', 'Please fill in both fields');
      return;
    }

    console.log('Attempting login with:', cleanEmail);

    // 🚀 Attempt to sign in directly (no email verification)
    const { data, error } = await supabase.auth.signInWithPassword({
      email: cleanEmail,
      password: cleanPassword,
    });

    if (error) {
      console.log('Login error:', error);
      Alert.alert('Login Failed', error.message);
      return;
    }

    const user = data?.user;
    if (!user) {
      Alert.alert('Error', 'No user data returned from Supabase.');
      return;
    }

    console.log('User logged in:', user.email);

    // 🧱 Ensure profile exists (auto-create if missing)
    const { data: existingProfile, error: checkError } = await supabase
      .from('profiles')
      .select('id')
      .eq('id', user.id)
      .maybeSingle();

    if (checkError) {
      console.error('Error checking profile:', checkError);
    }

    if (!existingProfile) {
      console.log('No profile found — creating new profile...');
      const { error: insertError } = await supabase.from('profiles').insert({
        id: user.id,
        email: user.email,
        display_name: '',
      });

      if (insertError) {
        console.error('Error creating profile:', insertError);
      } else {
        console.log('Profile created successfully');
      }
    }

    // ✅ Direct login success
    Alert.alert('Success', 'Login successful!');
    router.replace('/(tabs)/translate'); // make sure this route exists
  };

  return (
    <ImageBackground
      source={require('../../assets/images/LoginSignUpBG.png')}
      className="flex-1 justify-center items-center p-8"
      resizeMode="cover"
    >
      <View className="absolute inset-0 bg-black opacity-40" />

      <View className="relative w-full max-w-sm p-8 rounded-3xl">
        <Text className="text-4xl font-bold text-black mb-10 text-center">Welcome Back</Text>

        <TextInput
          className="w-full border-2 border-accent rounded-lg p-4 mb-4 text-black text-lg font-bold bg-neutral"
          placeholder="Email"
          placeholderTextColor="#444444"
          autoCapitalize="none"
          value={email}
          onChangeText={setEmail}
        />
        <TextInput
          className="w-full border-2 border-accent rounded-lg p-4 mb-4 text-black text-lg font-bold bg-neutral"
          placeholder="Password"
          placeholderTextColor="#444444"
          secureTextEntry
          value={password}
          onChangeText={setPassword}
        />

        <TouchableOpacity
          onPress={handleLogin}
          className="w-full bg-accent rounded-full py-4 items-center mt-4"
        >
          <Text className="text-white text-lg font-bold">Log In</Text>
        </TouchableOpacity>

        <View className="mt-6 flex-row items-center justify-center">
          <Text className="text-black">Don't have an account? </Text>
          <TouchableOpacity onPress={() => router.replace('/(stack)/SignUp')}>
            <Text className="text-highlight font-bold">Sign Up</Text>
          </TouchableOpacity>
        </View>
      </View>
    </ImageBackground>
  );
};

export default Login;

const styles = StyleSheet.create({});
