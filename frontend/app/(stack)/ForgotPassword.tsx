import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  TextInput,
  ImageBackground,
  Alert,
} from 'react-native';
import React, { useState } from 'react';
import { supabase } from '../../src/supabaseClient';
import { router } from 'expo-router';
import { ChevronLeft } from 'lucide-react-native';
import Message, { MessageType } from '../../components/Message';
import Animated, { FadeInUp, FadeInDown } from 'react-native-reanimated';

const ForgotPassword = () => {
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState<MessageType>('error');

  const showStatus = (msg: string, type: MessageType) => {
    setMessage(msg);
    setMessageType(type);
    setTimeout(() => setMessage(''), 5000);
  };

  const handleSendInstruction = async () => {
    const cleanEmail = email.trim().toLowerCase();

    if (!cleanEmail) {
      showStatus('Please enter your email address.', 'warning');
      return;
    }

    setLoading(true);

    try {
      // Request password reset. Since your template uses {{ .Token }}, this sends an OTP.
      const { error } = await supabase.auth.resetPasswordForEmail(cleanEmail);

      if (error) throw error;

      showStatus('Code sent! Check your email.', 'success');
      
      // Navigate to the verification screen passing the email
      setTimeout(() => {
        router.push({
          pathname: '/(stack)/verify-reset',
          params: { email: cleanEmail }
        });
      }, 1000);

    } catch (error: any) {
      showStatus(error.message || 'Failed to send code.', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleGoBack = () => {
    if (router.canGoBack()) router.back();
    else router.replace('/(stack)/Login');
  };

  return (
    <ImageBackground
      source={require('../../assets/images/LoginSignUpBG.png')}
      className="flex-1 justify-center items-center"
      resizeMode="cover"
    >
      <View className="absolute inset-0 bg-black opacity-40" />

      <TouchableOpacity
        onPress={handleGoBack}
        className="absolute top-12 left-8 p-2 rounded-full bg-white/80 z-10"
      >
        <ChevronLeft color="#1A1A1A" size={28} />
      </TouchableOpacity>

      <Message message={message} type={messageType} onClose={() => setMessage('')} />

      <Animated.View
        entering={FadeInUp.duration(700).delay(150)}
        className="relative w-full max-w-sm p-8 rounded-3xl bg-white/80"
      >
        <Animated.Text
          entering={FadeInDown.delay(200).duration(600)}
          className="text-3xl text-black mb-4 text-center font-audiowide"
        >
          Forgot Password?
        </Animated.Text>

        <Animated.Text
          entering={FadeInUp.delay(250).duration(600)}
          className="text-gray-600 mb-8 text-center font-fredoka text-base"
        >
          Enter your email address and we'll send you a code to reset your password.
        </Animated.Text>

        <Animated.View entering={FadeInUp.delay(300).duration(600)}>
          <TextInput
            className="w-full border-2 border-accent rounded-lg p-4 mb-4 text-black text-lg font-montserrat-semibold bg-neutral"
            placeholder="Email"
            placeholderTextColor="#444444"
            autoCapitalize="none"
            value={email}
            onChangeText={setEmail}
            selectionColor="#FFAB7B"
            keyboardType="email-address"
          />
        </Animated.View>

        <Animated.View entering={FadeInUp.delay(400).duration(600)}>
          <TouchableOpacity
            onPress={handleSendInstruction}
            disabled={loading}
            className={`w-full bg-accent rounded-full py-4 items-center mt-2 shadow-lg ${loading ? 'opacity-70' : ''}`}
          >
            <Text className="text-white text-lg font-audiowide">
              {loading ? 'Sending...' : 'Send Code'}
            </Text>
          </TouchableOpacity>
        </Animated.View>

      </Animated.View>
    </ImageBackground>
  );
};

export default ForgotPassword;

const styles = StyleSheet.create({});