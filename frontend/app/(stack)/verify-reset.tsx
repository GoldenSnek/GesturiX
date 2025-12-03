import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ImageBackground,
  StyleSheet,
  Dimensions,
  Keyboard,
  TouchableWithoutFeedback
} from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { ChevronLeft, Eye, EyeOff } from 'lucide-react-native';
import Message, { MessageType } from '../../components/Message';
import Animated, { FadeInUp, FadeInDown, Layout } from 'react-native-reanimated';
import { supabase } from '../../src/supabaseClient';

const { width } = Dimensions.get('window');
const BOX_SIZE = width / 8;

export default function VerifyReset() {
  const { email } = useLocalSearchParams<{ email: string }>();
  const router = useRouter();
  
  const [mode, setMode] = useState<'otp' | 'password'>('otp');

  // OTP State
  const [code, setCode] = useState('');
  const [resendTimer, setResendTimer] = useState(45);
  const [canResend, setCanResend] = useState(false);

  // Password State
  const [newPassword, setNewPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);

  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState<MessageType>('error');

  const inputRef = useRef<TextInput>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Timer for Resend Code
  useEffect(() => {
    if (resendTimer > 0 && mode === 'otp') {
      timerRef.current = setTimeout(() => setResendTimer((prev) => prev - 1), 1000);
    } else {
      setCanResend(true);
    }
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, [resendTimer, mode]);

  const showStatus = (msg: string, type: MessageType) => {
    setMessage(msg);
    setMessageType(type);
    setTimeout(() => setMessage(''), 5000);
  };

  // Step 1: Verify the OTP Code
  const handleVerifyCode = async () => {
    if (!code || code.length !== 6) {
      showStatus('Please enter a valid 6-digit code.', 'warning');
      return;
    }
    setLoading(true);

    // Use 'recovery' type for password reset flows
    const { error } = await supabase.auth.verifyOtp({
      email,
      token: code,
      type: 'recovery',
    });

    setLoading(false);

    if (error) {
      showStatus(error.message || 'Invalid code.', 'error');
    } else {
      showStatus('Code verified!', 'success');
      setMode('password');
    }
  };

  // Step 2: Reset the Password
  const handleResetPassword = async () => {
    if (newPassword.length < 6) {
      showStatus('Password must be at least 6 characters.', 'warning');
      return;
    }
    setLoading(true);

    const { error } = await supabase.auth.updateUser({
      password: newPassword
    });

    if (error) {
      setLoading(false);
      showStatus(error.message, 'error');
    } else {
      // Sign out to ensure they log in manually with the new password
      await supabase.auth.signOut();
      
      setLoading(false);
      showStatus('Password updated! Please log in.', 'success');
      
      setTimeout(() => {
        router.replace('/(stack)/Login');
      }, 1500);
    }
  };

  const handleResendCode = async () => {
    if (!canResend) return;
    setCanResend(false);
    setResendTimer(45);
    const { error } = await supabase.auth.resetPasswordForEmail(email);
    if (error) showStatus(error.message, 'error');
    else showStatus('Code resent!', 'success');
  };

  return (
    <ImageBackground
      source={require('../../assets/images/LoginSignUpBG.png')}
      className="flex-1 justify-center items-center"
      resizeMode="cover"
    >
      <View className="absolute inset-0 bg-black opacity-40" />

      <TouchableOpacity
        onPress={() => router.back()}
        className="absolute top-12 left-8 p-2 rounded-full bg-white/80 z-10"
      >
        <ChevronLeft color="#1A1A1A" size={28} />
      </TouchableOpacity>

      <Message message={message} type={messageType} onClose={() => setMessage('')} />

      <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
        <Animated.View
          layout={Layout.springify()}
          entering={FadeInUp.duration(700).delay(150)}
          className="w-[90%] max-w-md p-8 rounded-3xl bg-white/80 items-center"
        >
          
          {/* HEADER TEXT */}
          <Animated.Text
            key={mode} // Re-animate on mode change
            entering={FadeInDown.delay(100).duration(500)}
            className="text-3xl text-black mb-2 text-center font-audiowide"
          >
            {mode === 'otp' ? 'Verify Code' : 'New Password'}
          </Animated.Text>

          <Animated.Text
            entering={FadeInUp.delay(200).duration(600)}
            className="text-gray-600 mb-8 text-center font-fredoka text-base"
          >
            {mode === 'otp' ? (
              <>
                Enter the 6-digit code sent to{'\n'}
                <Text className="font-fredoka-semibold text-accent">{email}</Text>
              </>
            ) : (
              'Please enter your new password below.'
            )}
          </Animated.Text>

          {/* OTP INPUT MODE */}
          {mode === 'otp' && (
            <>
              <Animated.View entering={FadeInUp.delay(300)} className="w-full mb-8 items-center justify-center">
                <TextInput
                  ref={inputRef}
                  value={code}
                  onChangeText={(text) => /^\d*$/.test(text) && setCode(text)}
                  maxLength={6}
                  keyboardType="number-pad"
                  style={styles.hiddenInput}
                  autoFocus
                />
                <View style={styles.otpContainer}>
                  {[0, 1, 2, 3, 4, 5].map((index) => (
                    <TouchableOpacity
                      key={index}
                      activeOpacity={1}
                      onPress={() => inputRef.current?.focus()}
                      style={[
                        styles.otpBox,
                        {
                          borderColor: code.length === index ? '#FF6B00' : '#D3DEE8',
                          backgroundColor: code.length === index ? 'rgba(255, 107, 0, 0.1)' : 'white',
                        }
                      ]}
                    >
                      <Text className="text-2xl font-montserrat-bold text-black">
                        {code[index] || ''}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>
              </Animated.View>

              <TouchableOpacity
                onPress={handleVerifyCode}
                disabled={loading}
                className="w-full bg-accent rounded-full py-4 items-center shadow-lg"
              >
                <Text className="text-white text-lg font-audiowide">
                  {loading ? 'Verifying...' : 'Verify Code'}
                </Text>
              </TouchableOpacity>

              <View className="mt-6 flex-row">
                <Text className="text-gray-600 font-fredoka">Didn't receive code? </Text>
                <TouchableOpacity onPress={handleResendCode} disabled={!canResend}>
                  <Text className="font-fredoka-semibold" style={{ color: canResend ? '#FF6B00' : '#9CA3AF' }}>
                    {canResend ? 'Resend' : `Resend (${resendTimer}s)`}
                  </Text>
                </TouchableOpacity>
              </View>
            </>
          )}

          {/* NEW PASSWORD MODE */}
          {mode === 'password' && (
            <>
              <Animated.View entering={FadeInUp.delay(300)} className="w-full mb-4">
                <View className="w-full border-2 border-accent rounded-lg bg-neutral flex-row items-center">
                  <TextInput
                    className="flex-1 p-4 text-black text-lg font-montserrat-semibold"
                    placeholder="New Password"
                    placeholderTextColor="#444444"
                    secureTextEntry={!showPassword}
                    value={newPassword}
                    onChangeText={setNewPassword}
                    selectionColor="#FFAB7B"
                  />
                  <TouchableOpacity onPress={() => setShowPassword(!showPassword)} style={{ paddingRight: 16 }}>
                    {showPassword ? <EyeOff color="#0D47A1" size={22} /> : <Eye color="#0D47A1" size={22} />}
                  </TouchableOpacity>
                </View>
              </Animated.View>

              <TouchableOpacity
                onPress={handleResetPassword}
                disabled={loading}
                className="w-full bg-accent rounded-full py-4 items-center shadow-lg mt-4"
              >
                <Text className="text-white text-lg font-audiowide">
                  {loading ? 'Updating...' : 'Set Password'}
                </Text>
              </TouchableOpacity>
            </>
          )}

        </Animated.View>
      </TouchableWithoutFeedback>
    </ImageBackground>
  );
}

const styles = StyleSheet.create({
  hiddenInput: {
    position: 'absolute',
    width: '100%',
    height: '100%',
    opacity: 0,
    zIndex: 20,
  },
  otpContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '100%',
    paddingHorizontal: 2,
  },
  otpBox: {
    width: BOX_SIZE,
    height: 64,
    borderRadius: 8,
    borderWidth: 2,
    alignItems: 'center',
    justifyContent: 'center',
  },
});