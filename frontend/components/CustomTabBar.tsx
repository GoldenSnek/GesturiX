import React, { useEffect, useRef } from 'react';
import { View, TouchableOpacity, Text, Image, Animated, Dimensions, Platform, ImageSourcePropType } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { BottomTabBarProps } from '@react-navigation/bottom-tabs';

// Define the type for our tab names
type TabName = 'translate' | 'compose' | 'learn' | 'profile';

const { width } = Dimensions.get('window'); // Get screen width for positioning

// --- TabButton Component with Jiggle Animation ---
interface TabButtonProps {
  route: BottomTabBarProps['state']['routes'][0];
  isFocused: boolean;
  onPress: () => void;
  options: any;
  tabItemWidth: number;
}

const TabButton: React.FC<TabButtonProps> = ({ route, isFocused, onPress, options, tabItemWidth }) => {
  // ✅ FIX: Ensure iconSource is defined. We use the 'translate' icon as a fallback.
  const iconSource: ImageSourcePropType = {
    'translate': require('../assets/images/Translate-icon.png'),
    'compose': require('../assets/images/Compose-icon.png'),
    'learn': require('../assets/images/Learn-icon.png'),
    'profile': require('../assets/images/Profile-icon.png'),
  }[route.name as TabName] || require('../assets/images/Translate-icon.png'); // Fallback ensures it's never null

  const label = options.title || route.name;

  // Animation value for subtle rotation
  const rotationAnim = useRef(new Animated.Value(0)).current; // Rotation: 0 to 1

  const iconTintColor = isFocused ? '#FF6B00' : '#F8F8F8'; // Direct colors for tintStyle
  const textClassName = isFocused ? 'text-accent' : 'text-secondary';

  useEffect(() => {
    if (isFocused) {
      // **Jiggle/Rotation Animation**: Quick left-right rotation
      rotationAnim.setValue(0);
      Animated.sequence([
        // Jiggle left
        Animated.timing(rotationAnim, { toValue: 1, duration: 100, useNativeDriver: true }),
        // Jiggle right
        Animated.timing(rotationAnim, { toValue: -1, duration: 100, useNativeDriver: true }),
        // Return to center
        Animated.timing(rotationAnim, { toValue: 0, duration: 100, useNativeDriver: true }),
      ]).start();
      
    } else {
      // Ensure unfocused tabs are reset
      rotationAnim.setValue(0);
    }
  }, [isFocused]);

  const rotate = rotationAnim.interpolate({
    inputRange: [-1, 0, 1],
    outputRange: ['-5deg', '0deg', '5deg'], // Jiggle range
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
        // Apply only the rotation animation for the jiggle effect
        style={{ transform: [{ rotate }] }}
      >
        <Image 
          source={iconSource} 
          className="w-6 h-6" 
          style={{ tintColor: iconTintColor }} 
        />
        {isFocused && (
          // FOCUSED: Show Text (Accent)
          <Text 
            className={`text-xs font-orbitron mt-1 ${textClassName}`}
            numberOfLines={1}
          >
            {label}
          </Text>
        )}
      </Animated.View>
      {/* For unfocused tabs, we keep the icon static inside the animated view 
        which now only applies rotation on focus.
        We no longer need a separate rendering view. 
      */}
    </TouchableOpacity>
  );
};
// --- End TabButton Component ---


export default function CustomTabBar({ state, descriptors, navigation }: BottomTabBarProps) {
  const insets = useSafeAreaInsets();
  
  // NOTE: Keeping the translateX animation logic, though it's not currently used
  // in the visual appearance, as it was in the original code.
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
      // Main floating container
      className={`absolute bottom-0 left-0 right-0 items-center`}
      style={{
        // Add extra padding for iPhone X/notch devices for a 'floating' effect
        paddingBottom: insets.bottom + (Platform.OS === 'android' ? 0 : 20), 
      }}
    >
      <View
        // The oval-shaped container for the tabs
        className="flex-row items-center justify-around bg-darkbg/80 rounded-full px-2 py-2 shadow-xl"
        style={{
          width: width * 0.9, 
          height: 65,
          // Using a dark shadow to match the dark bar background
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