// File: frontend/components/CustomTabBar.tsx
import React, { useEffect, useRef } from 'react';
import { View, TouchableOpacity, Text, Image, Animated, Dimensions, Platform, ImageSourcePropType } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { BottomTabBarProps } from '@react-navigation/bottom-tabs';
import { MaterialIcons } from '@expo/vector-icons'; // ✅ Added for Quiz icon

// Define the type for our tab names
type TabName = 'translate' | 'learn' | 'quiz' | 'profile';

const { width } = Dimensions.get('window'); 

interface TabButtonProps {
  route: BottomTabBarProps['state']['routes'][0];
  isFocused: boolean;
  onPress: () => void;
  options: any;
  tabItemWidth: number;
}

const TabButton: React.FC<TabButtonProps> = ({ route, isFocused, onPress, options, tabItemWidth }) => {
  // Label comes from options.title (which we set to "Quiz" in _layout.tsx)
  const label = options.title || route.name;

  const rotationAnim = useRef(new Animated.Value(0)).current; 
  const iconTintColor = isFocused ? '#FF6B00' : '#F8F8F8'; 
  const textClassName = isFocused ? 'text-accent' : 'text-secondary';

  // ✅ Define icon source for image-based tabs only
  const getIconSource = (): ImageSourcePropType | null => {
    switch (route.name) {
      case 'translate': return require('../assets/images/Translate-icon.png');
      case 'learn': return require('../assets/images/Learn-icon.png');
      case 'profile': return require('../assets/images/Profile-icon.png');
      default: return null; // compose/quiz handled via MaterialIcons
    }
  };
  
  const iconSource = getIconSource();

  useEffect(() => {
    if (isFocused) {
      rotationAnim.setValue(0);
      Animated.sequence([
        Animated.timing(rotationAnim, { toValue: 1, duration: 100, useNativeDriver: true }),
        Animated.timing(rotationAnim, { toValue: -1, duration: 100, useNativeDriver: true }),
        Animated.timing(rotationAnim, { toValue: 0, duration: 100, useNativeDriver: true }),
      ]).start();
    } else {
      rotationAnim.setValue(0);
    }
  }, [isFocused]);

  const rotate = rotationAnim.interpolate({
    inputRange: [-1, 0, 1],
    outputRange: ['-5deg', '0deg', '5deg'],
  });

  return (
    <TouchableOpacity
      key={route.key}
      accessibilityRole="button"
      accessibilityState={isFocused ? { selected: true } : {}}
      accessibilityLabel={options.tabBarAccessibilityLabel}
      onPress={onPress}
      className={`flex-1 justify-center items-center h-full ${isFocused ? 'py-1' : ''}`}
      style={{ width: tabItemWidth }}
    >
      <Animated.View 
        className="flex-col items-center justify-center"
        style={{ transform: [{ rotate }] }}
      >
        {/* ✅ Conditional Rendering: MaterialIcons for 'compose' (Quiz), Image for others */}
        {route.name === 'quiz' ? (
          <MaterialIcons name="school" size={26} color={iconTintColor} />
        ) : (
          <Image 
            source={iconSource || require('../assets/images/Translate-icon.png')} 
            className="w-6 h-6" 
            style={{ tintColor: iconTintColor }} 
          />
        )}

        {isFocused && (
          <Text 
            className={`text-xs font-orbitron mt-1 ${textClassName}`}
            numberOfLines={1}
          >
            {label}
          </Text>
        )}
      </Animated.View>
    </TouchableOpacity>
  );
};

export default function CustomTabBar({ state, descriptors, navigation }: BottomTabBarProps) {
  const insets = useSafeAreaInsets();
  const translateX = useRef(new Animated.Value(0)).current; 
  const tabItemWidth = useRef(width / state.routes.length).current;

  useEffect(() => {
    const toValue = state.index * tabItemWidth;
    Animated.spring(translateX, {
      toValue,
      stiffness: 1000,
      damping: 50,
      mass: 1,
      useNativeDriver: true,
    }).start();
  }, [state.index, translateX, tabItemWidth]);

  return (
    <View
      className={`absolute bottom-0 left-0 right-0 items-center`}
      style={{
        paddingBottom: insets.bottom + (Platform.OS === 'android' ? 0 : 20), 
      }}
    >
      <View
        className="flex-row items-center justify-around bg-darkbg/80 rounded-full px-2 py-2 shadow-xl"
        style={{
          width: width * 0.9, 
          height: 65,
          shadowColor: '#1a1a1a', 
          shadowOffset: { width: 0, height: 6 },
          shadowOpacity: 0.9,
          shadowRadius: 10,
          elevation: 15,
        }}
      >
        {state.routes.map((route, index) => {
          const { options } = descriptors[route.key];
          const isFocused = state.index === index;

          const handlePress = () => {
            const event = navigation.emit({
              type: 'tabPress',
              target: route.key,
              canPreventDefault: true,
            });

            if (!isFocused && !event.defaultPrevented) {
              navigation.navigate(route.name, route.params);
            }
          };
          
          return (
            <TabButton
                key={route.key}
                route={route}
                isFocused={isFocused}
                onPress={handlePress}
                options={options}
                tabItemWidth={tabItemWidth}
            />
          );
        })}
      </View>
    </View>
  );
}