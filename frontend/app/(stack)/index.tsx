import { StyleSheet, Text, View, TouchableOpacity, ImageBackground, Image, ScrollView, Dimensions } from 'react-native';
import React from 'react';
import { router } from 'expo-router';
// We only keep basic Animated imports for the initial splash screen entrance effects
import Animated, { 
  FadeInUp, FadeInLeft, FadeInRight,
  useSharedValue, useAnimatedScrollHandler, useAnimatedStyle, withRepeat, withTiming, Easing, interpolate
} from 'react-native-reanimated';
import { BlurView } from 'expo-blur'; 

const windowHeight = Dimensions.get('window').height;

// --- TypeScript Interfaces for Props (Simplified) ---
interface FeatureItemProps {
  icon: string;
  title: string;
  description: string;
}

interface InfoBlockProps {
  title: string;
  content: string;
}
// ---------------------------------------

// --- Feature Item Component (Updated with dark background and light text) ---
const FeatureItem = ({ icon, title, description }: FeatureItemProps) => (
  <View className="w-full bg-black/60 p-6 rounded-2xl shadow-xl shadow-black/20 mb-6 border border-accent/50">
    <View className="flex-row items-center mb-2">
      <Text className="text-3xl mr-3">{icon}</Text>
      <Text className="text-xl font-montserrat-bold text-white">{title}</Text>
    </View>
    <Text className="text-base font-montserrat-regular text-gray-200">{description}</Text>
  </View>
);

// --- Info Block Component (Updated with accent border) ---
const InfoBlock = ({ title, content }: InfoBlockProps) => (
  <View className="w-full max-w-sm p-8 rounded-3xl bg-white/80 border-2 border-accent shadow-xl mb-8">
    {/* Text color maintained for light background contrast */}
    <Text className="text-2xl font-audiowide mb-3 text-center text-gray-800">{title}</Text>
    <Text className="text-lg text-center text-gray-600 font-montserrat-regular">{content}</Text>
  </View>
);

const LandingPage = () => {
  const scrollY = useSharedValue(0);

  // Animated Scroll Handler to track vertical offset
  const scrollHandler = useAnimatedScrollHandler((event) => {
    scrollY.value = event.contentOffset.y;
  });

  // Animated style for the pulsating down arrow
  const arrowAnimatedStyle = useAnimatedStyle(() => {
    // Fade out completely when scrolled past 50px
    const opacity = interpolate(
      scrollY.value,
      [0, 50],
      [1, 0]
    );

    // Pulsating scale animation
    const scale = withRepeat(
      withTiming(1.2, { duration: 800, easing: Easing.inOut(Easing.ease) }),
      -1, // -1 means infinite repeat
      true // Reverse the animation on each repeat
    );

    return {
      opacity,
      transform: [{ scale: scale }],
    };
  });


  return (
    <ImageBackground
      source={require('../../assets/images/LoginSignUpBG.png')} // main static background
      style={{ flex: 1 }}
      resizeMode="cover"
    >
      {/* Dark overlay for contrast on the entire page */}
      <View className="absolute inset-0 bg-black opacity-30" />
      
      {/* Use Animated.ScrollView for Reanimated scroll handler */}
      <Animated.ScrollView 
        style={{ flex: 1 }} 
        contentContainerStyle={{ flexGrow: 1, alignItems: 'center' }}
        onScroll={scrollHandler} // Attach the handler
        scrollEventThrottle={16} // Standard practice for smooth tracking
      >
        
        {/* SECTION 1: Top Landing Section (Full Viewport Height) */}
        <View style={{ height: windowHeight, width: '100%' }}>
          <ImageBackground
            source={require('../../assets/images/LandingPageBG.png')}
            style={{ flex: 1, justifyContent: 'space-between', paddingHorizontal: 32, paddingTop: 16, paddingBottom: 32 }}
            resizeMode="cover"
          >
            {/* Dark overlay for contrast on the landing image */}
            <View className="absolute inset-0 bg-black opacity-50" />

            {/* Logo (Removed mt-8 to allow it to sit lower and naturally space out from buttons) */}
            <Animated.View 
              entering={FadeInUp.duration(2000).delay(500)}
              className="z-10 items-center" 
            >
              <Image
                source={require('../../assets/images/GesturiX-motto-color.png')}
                style={{ width: 320, height: 320 }}
                resizeMode="contain"
              />
            </Animated.View>

            {/* Buttons (Positioned lower with mb-8) and Scroll Indicator */}
            <View className="z-10 w-full mb-8 items-center">
              {/* Get Started */}
              <Animated.View entering={FadeInRight.duration(500).delay(1500)} className="w-full">
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
              <Animated.View entering={FadeInLeft.duration(500).delay(2000)} className="w-full">
                <TouchableOpacity
                  onPress={() => router.push('/(stack)/Login')}
                  className="w-full border-2 border-accent rounded-full py-4 items-center shadow-lg shadow-black/40 bg-black/20"
                >
                  <Text className="text-accent text-lg font-montserrat-bolditalic">
                    Log In
                  </Text>
                </TouchableOpacity>
              </Animated.View>

              {/* Pulsating Scroll Down Indicator (Pushed further down with mt-12) */}
              <Animated.View style={arrowAnimatedStyle} className="">
                <Text className="text-accent text-3xl">⇩</Text>
              </Animated.View>
            </View>
          </ImageBackground>
        </View>

        {/* SECTION 2: Features & Highlights - ACCENT BACKGROUND, DARK/ACCENT TITLE, DARK CONTAINERS */}
        <View className="w-full items-center py-16 px-6 bg-accent/20">
          <Text 
            // Title changed to use the accent color for a dark, prominent look
            className="text-4xl font-audiowide mb-10 text-center text-gray-800 shadow-md shadow-black/50"
          >
            Features & Highlights
          </Text>
          
          <View className="w-full max-w-lg items-center">
            <FeatureItem 
              icon="🌍"
              title="Real-Time Translation"
              description="Instantly translate spoken and written words as you interact, breaking down communication barriers globally."
            />
            <FeatureItem 
              icon="📚"
              title="Learn Through Videos and Tutorials"
              description="Access an integrated library of video lessons and guides to quickly master new skills and app features."
            />
            <FeatureItem 
              icon="👤"
              title="Customizable Profile"
              description="Personalize your user experience, control privacy settings, and track your progress in one dedicated area."
            />
            <FeatureItem 
              icon="✨"
              title="Plus, Much More..."
              description="We're constantly working on future updates like spatial computing integration and advanced personalization!"
            />
          </View>
        </View>

        {/* SECTION 3: More Info & Mission - LIGHTER BACKGROUND, ACCENT BORDERS */}
        <View className="w-full items-center py-16 px-6 bg-white/10">
          <Text
            className="text-4xl font-audiowide mb-10 text-center text-gray-800 shadow-md shadow-black/10"
          >
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

        {/* SECTION 4: Footer (Static content) */}
        <View className="w-full items-center py-12 px-6 bg-black/80 border-t-2 border-accent">
          <View className="w-full max-w-lg items-center">
            <Text className="text-white text-3xl font-audiowide mb-4">GesturiX</Text>
            
            <View className="flex-row justify-center space-x-8 mb-6">
              <TouchableOpacity onPress={() => console.log('Privacy')}>
                <Text className="text-highlight text-base font-montserrat-regular">Privacy Policy   </Text>
              </TouchableOpacity>
              <TouchableOpacity onPress={() => console.log('Terms')}>
                <Text className="text-highlight text-base font-montserrat-regular">   Terms of Service   </Text>
              </TouchableOpacity>
              <TouchableOpacity onPress={() => console.log('Contact')}>
                <Text className="text-highlight text-base font-montserrat-regular">   Contact Us</Text>
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