import { StyleSheet, Text, View, TouchableOpacity, ImageBackground, Image, ScrollView, Dimensions } from 'react-native';
import React from 'react';
import { router } from 'expo-router';
import Animated, { 
  FadeInUp, FadeInLeft, FadeInRight,
  useSharedValue, useAnimatedScrollHandler, useAnimatedStyle, 
  interpolate, FadeInDown
} from 'react-native-reanimated';

const windowHeight = Dimensions.get('window').height;

// --- TypeScript Interfaces for Props (Simplified) ---
interface FeatureItemProps {
  title: string;
  description: string;
}

interface InfoBlockProps {
  title: string;
  content: string;
}
// ---------------------------------------

// --- Feature Item Component ---
const FeatureItem = ({ title, description }: FeatureItemProps) => (
  <View className="w-full bg-black/60 p-6 rounded-2xl shadow-xl shadow-black/20 mb-6 border border-accent/50">
    <Text className="text-xl font-montserrat-bold text-white mb-2">{title}</Text>
    <Text className="text-base font-montserrat-regular text-gray-200">{description}</Text>
  </View>
);

// --- Info Block Component ---
const InfoBlock = ({ title, content }: InfoBlockProps) => (
  <View className="w-full max-w-sm p-8 rounded-3xl bg-white/80 border-2 border-accent shadow-xl mb-8">
    <Text className="text-2xl font-audiowide mb-3 text-center text-gray-800">{title}</Text>
    <Text className="text-lg text-center text-gray-600 font-montserrat-regular">{content}</Text>
  </View>
);

const LandingPage = () => {
  const scrollY = useSharedValue(0);

  const scrollHandler = useAnimatedScrollHandler((event) => {
    scrollY.value = event.contentOffset.y;
  });

  const arrowAnimatedStyle = useAnimatedStyle(() => {
    const scrollVisibility = interpolate(
      scrollY.value,
      [0, 50],
      [1, 0]
    );

    return {
      opacity: scrollVisibility,
      transform: [{ scale: 1.0 }],
    };
  });


  return (
    <ImageBackground
      source={require('../../assets/images/LoginSignUpBG.png')}
      style={{ flex: 1 }}
      resizeMode="cover"
    >
      <View className="absolute inset-0 bg-black opacity-30" />
      
      <Animated.ScrollView 
        style={{ flex: 1 }} 
        contentContainerStyle={{ flexGrow: 1, alignItems: 'center' }}
        onScroll={scrollHandler}
        scrollEventThrottle={16}
      >
        
        {/* SECTION 1: Top Landing Section (Full Viewport Height) */}
        <View style={{ height: windowHeight, width: '100%' }}>
          <ImageBackground
            source={require('../../assets/images/LandingPageBG.png')}
            style={{ flex: 1, justifyContent: 'space-between', paddingHorizontal: 32, paddingTop: 16, paddingBottom: 32 }}
            resizeMode="cover"
          >
            <View className="absolute inset-0 bg-black opacity-50" />

            <Animated.View 
              entering={FadeInUp.duration(2000).delay(0)}
              className="z-10 items-center" 
            >
              <Image
                source={require('../../assets/images/GesturiX-motto-color.png')}
                style={{ width: 320, height: 320 }}
                resizeMode="contain"
              />
            </Animated.View>

            {/* Buttons and Scroll Indicator */}
            <View className="z-10 w-full mb-8 items-center">
              {/* Get Started */}
              <Animated.View entering={FadeInUp.duration(500).delay(1300)} className="w-full">
                <TouchableOpacity
                  onPress={() => router.push('/(stack)/SignUp')}
                  className="w-full bg-accent rounded-full py-4 mb-4 items-center shadow-lg shadow-black/40"
                >
                  <Text className="text-white text-lg font-montserrat-bolditalic">
                    Get Started
                  </Text>
                </TouchableOpacity>
              </Animated.View>

              {/* Log In */}
              <Animated.View entering={FadeInUp.duration(500).delay(1500)} className="w-full">
                <TouchableOpacity
                  onPress={() => router.push('/(stack)/Login')}
                  className="w-full border-2 border-accent rounded-full py-4 items-center shadow-lg shadow-black/40 bg-black/20"
                >
                  <Text className="text-accent text-lg font-montserrat-bolditalic">
                    Log In
                  </Text>
                </TouchableOpacity>
              </Animated.View>

              <Animated.View 
                entering={FadeInDown.duration(3000).delay(2500)} 
                className=""
              >
                <Animated.View style={arrowAnimatedStyle}>
                  <Text className="text-accent text-2xl">⇩</Text>
                </Animated.View>
              </Animated.View>
            </View>
          </ImageBackground>
        </View>

        {/* SECTION 2: Features & Highlights */}
        <View className="w-full items-center py-16 px-6 bg-accent/20">
          <Text className="text-4xl font-audiowide mb-10 text-center text-gray-800 shadow-md shadow-black/50">
            Features & Highlights
          </Text>
          
          <View className="w-full max-w-lg items-center">
            <FeatureItem 
              title="Real-Time Translation"
              description="Instantly translate spoken and written words as you interact, breaking down communication barriers globally."
            />
            <FeatureItem 
              title="Learn Through Videos and Tutorials"
              description="Access an integrated library of video lessons and guides to quickly master new skills and app features."
            />
            <FeatureItem 
              title="Customizable Profile"
              description="Personalize your user experience, control privacy settings, and track your progress in one dedicated area."
            />
            <FeatureItem 
              title="Plus, Much More..."
              description="We're constantly working on future updates like spatial computing integration and advanced personalization!"
            />
          </View>
        </View>

        {/* SECTION 3: More Info & Mission */}
        <View className="w-full items-center py-16 px-6 bg-white/10">
          <Text className="text-4xl font-audiowide mb-10 text-center text-gray-800 shadow-md shadow-black/10">
            Our Mission
          </Text>

          <View className="w-full max-w-lg items-center">
            <InfoBlock 
              title="Beyond the Screen"
              content="We believe interaction should be as fluid as thought. GesturiX aims to eliminate barriers between humanity and technology."
            />
            <InfoBlock 
              title="The Technology"
              content="Utilizing advanced computer vision and machine learning models, we translate real-time physical input into digital action."
            />
            <InfoBlock 
              title="Adaptable and Growing"
              content="Designed with flexibility, GesturiX continually adapts its interface and feature set to support the latest hardware and user needs."
            />
          </View>
        </View>

        {/* SECTION 4: Footer */}
        <View className="w-full items-center py-12 px-6 bg-black/80 border-t-2 border-accent">
          <View className="w-full max-w-lg items-center">
            <Text className="text-white text-3xl font-audiowide mb-4">GesturiX</Text>
            
            <View className="flex-row justify-center space-x-8 mb-6">
              <TouchableOpacity onPress={() => console.log('Privacy')}>
                <Text className="text-highlight text-base font-montserrat-regular">Privacy Policy   </Text>
              </TouchableOpacity>
              <TouchableOpacity onPress={() => console.log('Terms')}>
                <Text className="text-highlight text-base font-montserrat-regular">   Terms of Service   </Text>
              </TouchableOpacity>
              <TouchableOpacity onPress={() => console.log('Contact')}>
                <Text className="text-highlight text-base font-montserrat-regular">   Contact Us</Text>
              </TouchableOpacity>
            </View>
            
            <Text className="text-gray-500 text-center text-sm font-montserrat-regular">
              Gesturix is a modern sign language learning companion that is built with love and purpose to make learning sign language accessible for everyone UwU.
            </Text>

            <View className="mt-8 pt-4 border-t border-gray-700 w-full items-center">
              <Text className="text-white text-center text-lg">© 2025 GesturiX. All rights reserved.</Text>
            </View>
          </View>
        </View>

      </Animated.ScrollView>
    </ImageBackground>
  );
};

export default LandingPage;

const styles = StyleSheet.create({});