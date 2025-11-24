import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ImageBackground,
  StyleSheet,
  Dimensions,
} from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useAuth } from '../../src/AuthContext';
import { ChevronLeft } from 'lucide-react-native';
import Message, { MessageType } from '../../components/Message';
import Animated, { FadeInUp, FadeInDown } from 'react-native-reanimated';

const { width } = Dimensions.get('window');
const BOX_SIZE = width / 8; // Dynamic box size based on screen width

export default function VerifyCode() {
  const { email } = useLocalSearchParams<{ email: string }>();
  const [code, setCode] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState<MessageType>('error');
  
  // Resend Timer State
  const [resendTimer, setResendTimer] = useState(45);
  const [canResend, setCanResend] = useState(false);
  
  const { verifyOtp, resendOtp } = useAuth();
  const router = useRouter();
  const inputRef = useRef<TextInput>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);


  // Timer Logic
  useEffect(() => {
    if (resendTimer > 0) {
      timerRef.current = setTimeout(() => {
        setResendTimer((prev) => prev - 1);
      }, 1000);
    } else {
      setCanResend(true);
    }

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [resendTimer]);

  const showStatus = (msg: string, type: MessageType) => {
    setMessage(msg);
    setMessageType(type);
    setTimeout(() => setMessage(''), 5000);
  };

  const handleVerify = async () => {
    if (!code || code.length !== 6) {
      showStatus('Please enter a valid 6-digit code.', 'warning');
      return;
    }

    setLoading(true);
    const { error } = await verifyOtp(email, code);
    setLoading(false);

    if (error) {
      showStatus(error.message, 'error');
    } else {
      showStatus('Email verified successfully!', 'success');
      setTimeout(() => {
        router.replace('/(tabs)/translate');
      }, 1000);
    }
  };

  const handleResendCode = async () => {
    if (!canResend) return;

    // 1. Reset UI state immediately
    setCanResend(false);
    setResendTimer(45);

    // 2. Call Supabase to resend
    const { error } = await resendOtp(email);

    if (error) {
      // If error (e.g. rate limit), show error but timer continues to prevent spam
      showStatus(error.message, 'error');
    } else {
      showStatus('Verification code resent! Check your email.', 'success');
    }
  };

  const handleGoBack = () => {
    if (router.canGoBack()) router.back();
    else router.replace('/(stack)/SignUp');
  };

  return (
    <ImageBackground
      source={require('../../assets/images/LoginSignUpBG.png')}
      className="flex-1 justify-center items-center"
      resizeMode="cover"
    >
      {/* Dark Overlay */}
      <View className="absolute inset-0 bg-black opacity-40" />

      {/* Back Button */}
      <TouchableOpacity
        onPress={handleGoBack}
        className="absolute top-12 left-8 p-2 rounded-full bg-white/80 z-10"
      >
        <ChevronLeft color="#1A1A1A" size={28} />
      </TouchableOpacity>

      {/* Status Message */}
      <Message message={message} type={messageType} onClose={() => setMessage('')} />

      {/* Main Card */}
      <Animated.View
        entering={FadeInUp.duration(700).delay(150)}
        className="w-[90%] max-w-md p-8 rounded-3xl bg-white/80 items-center"
      >
        {/* Header */}
        <Animated.Text
          entering={FadeInDown.delay(200).duration(600)}
          className="text-3xl text-black mb-2 text-center font-audiowide"
        >
          Verify Email
        </Animated.Text>

        <Animated.Text
          entering={FadeInUp.delay(250).duration(600)}
          className="text-gray-600 mb-8 text-center font-fredoka text-base"
        >
          We sent a 6-digit code to{'\n'}
          <Text className="font-fredoka-semibold text-accent">{email}</Text>
        </Animated.Text>

        {/* OTP Input Section */}
        <Animated.View 
          entering={FadeInUp.delay(350).duration(600)}
          className="w-full mb-8 items-center justify-center"
        >
          {/* Invisible Input overlay to handle typing */}
          <TextInput
            ref={inputRef}
            value={code}
            onChangeText={(text) => {
              // Only allow numbers
              if (/^\d*$/.test(text)) {
                setCode(text);
              }
            }}
            maxLength={6}
            keyboardType="number-pad"
            style={styles.hiddenInput}
            autoFocus
          />

          {/* Visible Boxes */}
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

        {/* Verify Button */}
        <Animated.View entering={FadeInUp.delay(450).duration(600)} className="w-full">
          <TouchableOpacity
            onPress={handleVerify}
            disabled={loading}
            className="w-full bg-accent rounded-full py-4 items-center shadow-lg"
          >
            <Text className="text-white text-lg font-audiowide">
              {loading ? 'Verifying...' : 'Verify Account'}
            </Text>
          </TouchableOpacity>
        </Animated.View>

        {/* Resend Code with Timer */}
        <Animated.View entering={FadeInUp.delay(550).duration(600)} className="mt-6 flex-row">
          <Text className="text-gray-600 font-fredoka">Didn't receive a code? </Text>
          <TouchableOpacity 
            onPress={handleResendCode}
            disabled={!canResend}
          >
            <Text 
              className="font-fredoka-semibold"
              style={{ color: canResend ? '#FF6B00' : '#9CA3AF' }}
            >
              {canResend ? 'Resend' : `Resend (${resendTimer}s)`}
            </Text>
          </TouchableOpacity>
        </Animated.View>

      </Animated.View>
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
    // Small padding to prevent boxes from touching the container edges
    paddingHorizontal: 4, 
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