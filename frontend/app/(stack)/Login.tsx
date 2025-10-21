import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  TextInput,
  ImageBackground,
} from 'react-native';
import React, { useState } from 'react';
import { supabase } from '../../src/supabaseClient';
import { router } from 'expo-router';
import { Eye, EyeOff, ChevronLeft } from 'lucide-react-native';
import Message, { MessageType } from '../../components/Message';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState<MessageType>('error');

  const showStatus = (msg: string, type: MessageType) => {
    setMessage(msg);
    setMessageType(type);
    setTimeout(() => setMessage(''), 5000);
  };

  const showError = (msg: string) => showStatus(msg, 'error');
  const showWarning = (msg: string) => showStatus(msg, 'warning');
  const showSuccess = (msg: string) => showStatus(msg, 'success');

  const handleLogin = async () => {
    const cleanEmail = email.trim().toLowerCase();
    const cleanPassword = password.trim();

    if (!cleanEmail || !cleanPassword) {
      showWarning('Please enter both email and password.');
      return;
    }

    const { data, error } = await supabase.auth.signInWithPassword({
      email: cleanEmail,
      password: cleanPassword,
    });

    if (error) {
      showError(error.message || 'Login Failed. Please check your credentials.');
      return;
    }

    const user = data?.user;
    if (!user) {
      showError('User session could not be established.');
      return;
    }

    const { data: existingProfile, error: checkError } = await supabase
      .from('profiles')
      .select('id')
      .eq('id', user.id)
      .maybeSingle();

    if (checkError) console.error('Error checking profile:', checkError);

    if (!existingProfile) {
      const { error: insertError } = await supabase.from('profiles').insert({
        id: user.id,
        email: user.email,
        display_name: '',
      });
      if (insertError) console.error('Error creating profile:', insertError);
    }

    showSuccess('Login successful!');
    router.replace('/(tabs)/translate');
  };

  const handleGoBack = () => {
    if (router.canGoBack()) {
      router.back();
    } else {
      router.replace('/(stack)/SignUp');
    }
  };

  return (
    <ImageBackground
      source={require('../../assets/images/LoginSignUpBG.png')}
      className="flex-1 justify-center items-center p-8"
      resizeMode="cover"
    >
      <View className="absolute inset-0 bg-black opacity-40" />

      {/* 🆕 GO BACK BUTTON */}
      <TouchableOpacity
        onPress={handleGoBack}
        className="absolute top-12 left-8 p-2 rounded-full bg-white/80 z-10"
      >
        <ChevronLeft color="#1A1A1A" size={28} />
      </TouchableOpacity>

      <Message
        message={message}
        type={messageType}
        onClose={() => setMessage('')}
      />

      <View className="relative w-full max-w-sm p-8 rounded-3xl bg-white/80">
        {/* 🔹 TITLE — futuristic style */}
        <Text className="text-4xl text-black mb-10 text-center font-audiowide">
          Welcome Back
        </Text>

        {/* 🔹 Inputs — friendly rounded style */}
        <TextInput
          className="w-full border-2 border-accent rounded-lg p-4 mb-4 text-black text-lg font-montserrat-semibold bg-neutral"
          placeholder="Email"
          placeholderTextColor="#444444"
          autoCapitalize="none"
          value={email}
          onChangeText={setEmail}
        />

        <View className="w-full border-2 border-accent rounded-lg mb-4 bg-neutral flex-row items-center">
          <TextInput
            className="flex-1 p-4 text-black text-lg font-montserrat-semibold"
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

        {/* 🔹 Button — clean and bold tech style */}
        <TouchableOpacity
          onPress={handleLogin}
          className="w-full bg-accent rounded-full py-4 items-center mt-4"
        >
          <Text className="text-white text-lg font-audiowide">Log In</Text>
        </TouchableOpacity>

        <View className="mt-6 flex-row items-center justify-center">
          <Text className="text-black font-fredoka">Don't have an account? </Text>
          <TouchableOpacity onPress={() => router.replace('/(stack)/SignUp')}>
            <Text className="text-highlight font-fredoka-bold">Sign Up</Text>
          </TouchableOpacity>
        </View>
      </View>
    </ImageBackground>
  );
};

export default Login;

const styles = StyleSheet.create({});