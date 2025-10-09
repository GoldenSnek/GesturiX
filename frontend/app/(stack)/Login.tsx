import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  TextInput,
  ImageBackground,
  Alert, // Keeping Alert import, but minimizing usage
} from 'react-native';
import React, { useState } from 'react';
import { supabase } from '../../src/supabaseClient';
import { router } from 'expo-router';
import { Eye, EyeOff } from 'lucide-react-native';
import Message, { MessageType } from '../../components/Message'; 

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);

  // 👈 MESSAGE STATE
  const [message, setMessage] = useState(''); 
  const [messageType, setMessageType] = useState<MessageType>('error');

  // 🔹 Helper functions
  const showStatus = (msg: string, type: MessageType) => {
    setMessage(msg);
    setMessageType(type);
    // Automatically clear the message after 5 seconds
    setTimeout(() => setMessage(''), 5000); 
  };

  const showError = (msg: string) => showStatus(msg, 'error');
  const showWarning = (msg: string) => showStatus(msg, 'warning');
  const showSuccess = (msg: string) => showStatus(msg, 'success');

  const handleLogin = async () => {
    const cleanEmail = email.trim().toLowerCase();
    const cleanPassword = password.trim();

    // 👈 Replaced Alert with showError
    if (!cleanEmail || !cleanPassword) {
      showWarning('Please enter both email and password.');
      return;
    }

    console.log('Attempting login with:', cleanEmail);

    const { data, error } = await supabase.auth.signInWithPassword({
      email: cleanEmail,
      password: cleanPassword,
    });

    if (error) {
      console.log('Login error:', error);
      // 👈 Replaced Alert with showError
      // Supabase often returns a specific error, e.g., "Invalid login credentials"
      showError(error.message || 'Login Failed. Please check your credentials.');
      return;
    }

    const user = data?.user;
    if (!user) {
      // 👈 Replaced Alert with showError
      showError('User session could not be established.');
      return;
    }

    console.log('User logged in:', user.email);

    // --- Profile Check Logic (No changes needed for success) ---
    const { data: existingProfile, error: checkError } = await supabase
      .from('profiles')
      .select('id')
      .eq('id', user.id)
      .maybeSingle();

    if (checkError) console.error('Error checking profile:', checkError);

    if (!existingProfile) {
      // This case should ideally not happen for login, but if it does, 
      // we can use a warning or a technical error message.
      console.log('No profile found — creating new profile...');
      const { error: insertError } = await supabase.from('profiles').insert({
        id: user.id,
        email: user.email,
        display_name: '',
      });
      if (insertError) console.error('Error creating profile:', insertError);
      else console.log('Profile created successfully');
    }

    // 👈 Replaced Alert with showSuccess
    showSuccess('Login successful!');
    router.replace('/(tabs)/translate');
  };

  return (
    <ImageBackground
      source={require('../../assets/images/LoginSignUpBG.png')}
      className="flex-1 justify-center items-center p-8"
      resizeMode="cover"
    >
      <View className="absolute inset-0 bg-black opacity-40" />

      {/* 👈 RENDER THE MESSAGE COMPONENT */}
      <Message 
        message={message} 
        type={messageType}
        onClose={() => setMessage('')} 
      />

      <View className="relative w-full max-w-sm p-8 rounded-3xl bg-white/80">
        <Text className="text-4xl font-bold text-black mb-10 text-center">
          Welcome Back
        </Text>

        <TextInput
          className="w-full border-2 border-accent rounded-lg p-4 mb-4 text-black text-lg font-bold bg-neutral"
          placeholder="Email"
          placeholderTextColor="#444444"
          autoCapitalize="none"
          value={email}
          onChangeText={setEmail}
        />

        {/* Password input with toggle */}
        <View className="w-full border-2 border-accent rounded-lg mb-4 bg-neutral flex-row items-center">
          <TextInput
            className="flex-1 p-4 text-black text-lg font-bold"
            placeholder="Password"
            placeholderTextColor="#444444"
            secureTextEntry={!showPassword}
            value={password}
            onChangeText={setPassword}
          />
          <TouchableOpacity
            onPress={() => setShowPassword(!showPassword)}
            style={{ paddingRight: 16 }}
          >
            {showPassword ? (
              <EyeOff color="#0D47A1" size={22} />
            ) : (
              <Eye color="#0D47A1" size={22} />
            )}
          </TouchableOpacity>
        </View>

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