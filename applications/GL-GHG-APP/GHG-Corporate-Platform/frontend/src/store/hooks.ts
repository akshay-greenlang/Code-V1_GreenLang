/**
 * GL-GHG Corporate Platform - Typed Redux Hooks
 *
 * Provides pre-typed useDispatch and useSelector hooks
 * so components do not need to import RootState/AppDispatch directly.
 */

import { TypedUseSelectorHook, useDispatch, useSelector } from 'react-redux';
import type { AppDispatch, AppRootState } from './index';

/** Typed dispatch hook - use instead of plain useDispatch. */
export const useAppDispatch: () => AppDispatch = useDispatch;

/** Typed selector hook - use instead of plain useSelector. */
export const useAppSelector: TypedUseSelectorHook<AppRootState> = useSelector;
